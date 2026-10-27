"""
threshold_sensitivity.py
========================
Threshold sensitivity analysis cho CLEF framework.

==============================================================================
MỤC ĐÍCH
==============================================================================
Chứng minh tính bền vững (robustness) của CLEF khi thay đổi ngưỡng cảnh báo
anomaly detection threshold (percentile ∈ {90, 92, 95, 97, 99}).

Khắc phục triệt để:
  1. Normal pool ở Train đồng bộ theo đúng percentile đang khảo sát.
  2. Lấy mẫu phân tầng (stratified) trải đều từ điểm cực đoan nhất đến điểm
     sát ngưỡng cảnh báo, phản ánh đúng bản chất tập dị biệt ở từng ngưỡng.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep_cubems import (
    load_zone_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS, ZONE_IDS,
)
from anomaly_detection import build_detectors
from explainers import (
    explain_anomaly, get_predict_fn,
    compute_global_importance, compute_context_weights,
)
from metrics import compute_metrics
from walk_forward_cubems import build_folds_for_length

# ============================================================
# Config
# ============================================================

PERCENTILES = [90, 92, 95, 97, 99]
DEFAULT_ZONES = [
    "floor2_z1", 
    "floor2_z2", 
    "floor2_z4", 
    "floor5_z1", 
    "floor5_z4", 
    "floor5_z5"
]
DEFAULT_DETECTORS = ["IF"]
DEFAULT_METHODS = ["SHAP-CA", "LIME-std", "LIME-CA"]
TOP_K = 20       # Số lượng mẫu đại diện để giải thích
REPS = 10        # Số lần lặp để tính Stability
SEED = 42


# ============================================================
# Per-fold, per-percentile pipeline
# ============================================================

def run_fold_percentile(
    zone_id: str,
    X_scaled: np.ndarray,
    fold,
    detector_names: list[str],
    methods: list[str],
    percentile: int,
    top_k: int,
    reps: int,
    seed: int,
) -> list[dict]:
    """Chạy 1 fold với 1 percentile threshold."""
    train_end = fold.train_end
    val_end = fold.val_end
    test_end = fold.test_end

    X_train = X_scaled[:train_end]
    X_val = X_scaled[train_end:val_end]
    X_test = X_scaled[val_end:test_end]

    # 1. Fit detector CHỈ trên train
    detectors = build_detectors(
        X_train, detector_names, X_val=X_val, random_state=seed
    )

    rows = []

    for det_name, det in detectors.items():
        scores_all = det.score(X_scaled)
        scores_train = scores_all[:train_end]
        scores_val = scores_all[train_end:val_end]
        scores_test = scores_all[val_end:test_end]

        # 2. Khóa ngưỡng theta từ VALIDATION
        theta = float(np.percentile(scores_val, percentile))

        # 3. Đếm số lượng quan sát bị gắn cờ trong Test
        flagged_local = np.where(scores_test > theta)[0]
        n_flagged_test = int(len(flagged_local))

        # 4. ĐỒNG BỘ: Normal pool từ TRAIN theo đúng percentile đang xét
        theta_train = float(np.percentile(scores_train, percentile))
        train_indices = np.arange(0, train_end)
        normal_idx_global = train_indices[scores_train <= theta_train]

        # 5. LẤY MẪU PHÂN TẦNG: Đại diện toàn bộ dải bất thường vượt ngưỡng
        test_global_indices = np.arange(val_end, test_end)
        if n_flagged_test == 0:
            # Fallback nếu không có điểm nào vượt ngưỡng
            top_k_local = np.argsort(scores_test)[-top_k:][::-1]
        elif n_flagged_test <= top_k:
            # Lấy toàn bộ các điểm bị gắn cờ nếu số lượng ít hơn top_k
            order = np.argsort(scores_test[flagged_local])[::-1]
            top_k_local = flagged_local[order]
        else:
            # Sắp xếp các điểm gắn cờ theo chiều giảm dần của anomaly score
            sorted_flagged = flagged_local[np.argsort(scores_test[flagged_local])[::-1]]
            # Lấy đều K điểm trải dài từ điểm cực đoan nhất đến điểm sát ngưỡng theta
            stride_indices = np.linspace(0, n_flagged_test - 1, top_k, dtype=int)
            top_k_local = sorted_flagged[stride_indices]

        top_k_global = test_global_indices[top_k_local]

        # 6. Predict function
        predict_fn = get_predict_fn(det, det_name)

        # 7. Trọng số ngữ cảnh tính từ tập normal train tương ứng
        X_train_normal = X_scaled[normal_idx_global]
        global_importance = compute_global_importance(
            predict_fn, X_train_normal, n_repeats=5, seed=seed,
        )
        context_weights = compute_context_weights(
            global_importance, FEATURE_COLUMNS, CONTEXT_COLUMNS,
        )

        methods_for_det = list(methods)
        if det_name == "MLP":
            for m in ["DeepSHAP", "IG"]:
                if m not in methods_for_det:
                    methods_for_det.append(m)

        context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

        # 8. Giải thích các mẫu và tính toán 4 metrics
        for method in methods_for_det:
            phi_runs_per_anomaly = {}
            for g_idx in top_k_global:
                g_idx = int(g_idx)
                runs = []
                for rep in range(reps):
                    res_kwargs = {"seed": seed + rep}
                    if method in ("SHAP-CA", "LIME-CA"):
                        res_kwargs["context_weights"] = context_weights

                    res = explain_anomaly(
                        det, det_name,
                        X=X_scaled, idx=g_idx,
                        normal_idx=normal_idx_global,
                        method=method, device=zone_id,
                        context_idx=context_idx,
                        **res_kwargs,
                    )
                    runs.append(res.phi)
                phi_runs_per_anomaly[g_idx] = runs

            eval_pool_indices = np.arange(0, train_end)
            bundle = compute_metrics(
                predict_fn, X_scaled,
                top_k_global.tolist(),
                phi_runs_per_anomaly,
                eval_pool_idx=eval_pool_indices,
            )
            rows.append({
                "zone": zone_id,
                "fold": fold.idx,
                "percentile": percentile,
                "detector": det_name,
                "method": method,
                "theta": theta,
                "n_flagged_test": n_flagged_test,
                "n_explained": len(top_k_global),
                "n_normal_pool": len(normal_idx_global),
                "consistency": bundle.consistency,
                "stability": bundle.stability,
                "fidelity": bundle.fidelity,
                "sensitivity": bundle.sensitivity,
            })

    return rows


# ============================================================
# Main Execution
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zones", nargs="+", default=DEFAULT_ZONES, choices=ZONE_IDS)
    p.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--percentiles", nargs="+", type=int, default=PERCENTILES)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/threshold_sensitivity"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("THRESHOLD SENSITIVITY ANALYSIS (STRATIFIED)")
    print("=" * 78)
    print(f"Zones:       {args.zones}")
    print(f"Detectors:   {args.detectors}")
    print(f"Methods:     {args.methods}")
    print(f"Percentiles: {args.percentiles}")
    print(f"Top-K:       {args.top_k}")
    print(f"Reps:        {args.reps}")
    print(f"Output:      {args.output_dir}")
    print(f"\nProtocol: Threshold = percentile of VALIDATION scores")
    print(f"Sampling: Stratified across flagged range [extreme -> borderline]")

    t_start = time.perf_counter()
    all_rows = []

    for zone_id in args.zones:
        print(f"\n{'#'*78}")
        print(f"# ZONE: {zone_id}")
        print(f"{'#'*78}")

        df = load_zone_dataframe(zone_id, verbose=False)
        X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
        print(f"  Loaded: {X_raw.shape[0]} samples × {X_raw.shape[1]} features")

        folds = build_folds_for_length(len(X_raw))
        for fold in folds:
            print(f"\n  {'-'*74}")
            print(f"  Fold {fold.idx}: Train=[0:{fold.train_end}]  "
                  f"Val=[{fold.train_end}:{fold.val_end}]  "
                  f"Test=[{fold.val_end}:{fold.test_end}]")
            print(f"  {'-'*74}")

            # Fit scaler CHỈ trên train
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(X_raw[:fold.train_end])
            X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

            for pct in args.percentiles:
                t0 = time.perf_counter()
                rows = run_fold_percentile(
                    zone_id=zone_id,
                    X_scaled=X_scaled,
                    fold=fold,
                    detector_names=args.detectors,
                    methods=args.methods,
                    percentile=pct,
                    top_k=args.top_k,
                    reps=args.reps,
                    seed=args.seed,
                )
                all_rows.extend(rows)
                elapsed = time.perf_counter() - t0

                n_flagged = rows[0]["n_flagged_test"] if rows else 0
                n_exp = rows[0]["n_explained"] if rows else 0
                print(f"\n    Percentile {pct}%: n_flagged={n_flagged}  n_explained={n_exp}  ({elapsed:.1f}s)")
                for r in rows:
                    if r["method"] in args.methods:
                        print(f"      [{r['method']:9s}] "
                              f"cons={r['consistency']:.4f}  "
                              f"stab={r['stability']:.4f}  "
                              f"fid={r['fidelity']:.6f}  "
                              f"sens={r['sensitivity']:.4f}")

    # Lưu dữ liệu chi tiết
    df_all = pd.DataFrame(all_rows)
    per_cell_csv = args.output_dir / "per_cell_metrics.csv"
    df_all.to_csv(per_cell_csv, index=False)
    print(f"\n[Saved] Per-cell: {per_cell_csv}")

    # Báo cáo tổng hợp
    print(f"\n{'='*78}")
    print("THRESHOLD SENSITIVITY — MAIN REPORT (MEAN ACROSS FOLDS & ZONES)")
    print(f"{'='*78}")

    for method in args.methods:
        print(f"\n>>> METHOD: {method} <<<")
        sub = df_all[df_all["method"] == method]
        if sub.empty:
            continue

        pivot = sub.groupby("percentile").agg({
            "n_flagged_test": "mean",
            "consistency": "mean",
            "stability": "mean",
            "fidelity": "mean",
            "sensitivity": "mean",
        }).round(4)

        print(pivot.to_string())

        cons_vals = pivot["consistency"].values
        cons_min, cons_max = cons_vals.min(), cons_vals.max()
        cons_range = cons_max - cons_min
        rel_range = 100 * cons_range / (cons_max + 1e-12)

        print(f"  → Consistency range: {cons_min:.4f} – {cons_max:.4f}  "
              f"(relative spread: {rel_range:.2f}%)")

        if rel_range < 5:
            print("  → KẾT LUẬN: Highly Robust (<5% variation across operational cutoffs)")
        elif rel_range < 15:
            print("  → KẾT LUẬN: Moderately Robust (5-15% variation)")
        else:
            print("  → KẾT LUẬN: Sensitive to threshold (>15% variation)")

    print(f"\n{'='*78}")
    print("METHOD RANKING ACROSS PERCENTILES (Consistency Comparison)")
    print(f"{'='*78}")

    ranking = df_all.groupby(["percentile", "method"])["consistency"].mean().unstack()
    print(ranking.round(4).to_string())

    print(f"\n{'='*78}")
    print(f"Total runtime: {time.perf_counter() - t_start:.1f}s")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()