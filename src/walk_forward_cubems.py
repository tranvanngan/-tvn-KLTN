
"""
walk_forward_cubems.py
======================
Walk-forward cross-validation cho CUBEMS với tỷ lệ động thích ứng theo số mẫu thực tế.

Thiết kế Expanding Window (3 folds - Strictly Non-Overlapping Test Horizons):
    Fold 1: Train 0->45%, Val 45->55%, Test 55->70% (Bao phủ 70% dữ liệu đầu)
    Fold 2: Train 0->55%, Val 55->70%, Test 70->85% (Bao phủ 85% dữ liệu)
    Fold 3: Train 0->70%, Val 70->85%, Test 85->100% (Bao phủ 100% dữ liệu cuối)

Tất cả random seeds được khóa. Chạy 2 lần với cùng seed → bit-identical.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
    explain_anomaly,
    get_predict_fn,
    compute_global_importance,
    compute_context_weights,
)
from metrics import compute_metrics


# ============================================================
# Fold boundaries động theo tỷ lệ chuỗi thời gian liên tục
# ============================================================

# Cấu hình mới đảm bảo: Train -> Val -> Test liên tục trong từng Fold.
# Đồng thời tập Test của fold sau nối tiếp 100% ngay sau tập Test của fold trước.
FOLD_RATIOS = [
    # (train_end_ratio, val_end_ratio, test_end_ratio)
    (0.45, 0.55, 0.70),  # Fold 1: Train 45%, Val 10% (45-55), Test 15% (55-70)
    (0.55, 0.70, 0.85),  # Fold 2: Train 55%, Val 15% (55-70), Test 15% (70-85)
    (0.70, 0.85, 1.00),  # Fold 3: Train 70%, Val 15% (70-85), Test 15% (85-100)
]


@dataclass
class Fold:
    idx: int
    train_end: int
    val_end: int
    test_end: int

    def describe(self, timestamps):
        n_tr = self.train_end
        n_va = self.val_end - self.train_end
        n_te = self.test_end - self.val_end
        return (
            f"Train=[0:{self.train_end}] ({n_tr}h)  "
            f"Val=[{self.train_end}:{self.val_end}] ({n_va}h)  "
            f"Test=[{self.val_end}:{self.test_end}] ({n_te}h)  "
            f"| {timestamps[self.val_end].date()} -> "
            f"{timestamps[self.test_end - 1].date()}"
        )


def build_folds_for_length(n_total: int) -> list[Fold]:
    """Tạo fold boundaries chuỗi thời gian liên tục, các tập Test nối đuôi hoàn hảo."""
    folds = []
    for i, (tr_r, va_r, te_r) in enumerate(FOLD_RATIOS):
        tr_end = int(round(n_total * tr_r))
        va_end = int(round(n_total * va_r))
        
        # Đảm bảo fold cuối phủ hết toàn bộ mảng dữ liệu gốc
        if i == len(FOLD_RATIOS) - 1:
            te_end = n_total
        else:
            te_end = int(round(n_total * te_r))
            
        folds.append(Fold(idx=i + 1, train_end=tr_end, val_end=va_end, test_end=te_end))
    return folds


# ============================================================
# Per-fold pipeline 
# ============================================================

def run_fold(
    zone_id: str,
    X: np.ndarray,
    fold: Fold,
    detector_names: list[str],
    methods: list[str],
    top_k: int,
    reps: int,
    seed: int,
    verbose: bool = True,
) -> list[dict]:
    """Chạy 1 fold cho 1 zone với scaler fit CHỈ trên train và thresholding từ validation."""
    train_end, val_end, test_end = fold.train_end, fold.val_end, fold.test_end

    # 1. Fit scaler CHỈ trên train
    X_train_raw = X[:train_end]
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train_raw)

    # 2. Scale toàn bộ X và tách các tập
    X_scaled = np.clip(scaler.transform(X), 0.0, 1.0)
    X_train = X_scaled[:train_end]
    X_val = X_scaled[train_end:val_end]
    X_test = X_scaled[val_end:test_end]

    if verbose:
        print(f"\n    X_train: {X_train.shape}  "
              f"X_val: {X_val.shape}  "
              f"X_test: {X_test.shape}  "
              f"(test indices: {val_end}-{test_end-1})")

    # 3. Fit detectors với X_val truyền vào
    detectors = build_detectors(
        X_train, detector_names, X_val=X_val, random_state=seed
    )

    rows = []

    for det_name, det in detectors.items():
        # Score toàn bộ X_scaled
        scores_all = det.score(X_scaled)
        scores_train = scores_all[:train_end]
        scores_val = scores_all[train_end:val_end]
        scores_test = scores_all[val_end:test_end]

        # 4. Calibration: Ngưỡng lấy từ VALIDATION, normal pool lấy từ TRAIN
        theta = float(np.percentile(scores_val, 95))
        theta_train = float(np.percentile(scores_train, 95))

        # Normal pool: train indices có score <= theta_train
        train_indices = np.arange(0, train_end)
        normal_idx_global = train_indices[scores_train <= theta_train]

        # Predict function của detector
        predict_fn = get_predict_fn(det, det_name)

        # 5. Global feature importance từ training-normal data (No leakage)
        X_train_normal = X_scaled[normal_idx_global]

        global_importance = compute_global_importance(
            predict_fn,
            X_train_normal,
            n_repeats=5,
            seed=seed,
        )
        context_weights = compute_context_weights(
            global_importance,
            FEATURE_COLUMNS,
            CONTEXT_COLUMNS,
        )

        if verbose:
            print(
                f"        context_weights = "
                f"{ {k: round(v, 4) for k, v in context_weights.items()} }"
            )
            print(
                f"        global_importance = "
                f"{np.round(global_importance, 4)}"
            )

        # 6. Lấy mẫu Test Anomalies Phân tầng (Stratified Quantile Sampling)
        test_global_indices = np.arange(val_end, test_end)
        flagged_local = np.where(scores_test > theta)[0]

        if len(flagged_local) == 0:
            top_k_local = np.argsort(scores_test)[-top_k:][::-1]
        else:
            sorted_flagged = flagged_local[np.argsort(scores_test[flagged_local])[::-1]]
            stride = np.linspace(
                0, len(sorted_flagged) - 1, min(top_k, len(sorted_flagged)), dtype=int
            )
            top_k_local = sorted_flagged[stride]

        top_k_global = test_global_indices[top_k_local]

        if verbose:
            n_above = int((scores_test > theta).sum())
            print(f"\n      [{det_name}] theta_val={theta:.6f}  "
                  f"n_normal_train={len(normal_idx_global)}  "
                  f"n_test_above_theta={n_above}  "
                  f"n_explained={len(top_k_global)}")

        # MLP có thêm DeepSHAP, IG
        methods_for_det = list(methods)
        if det_name == "MLP":
            for m in ["DeepSHAP", "IG"]:
                if m not in methods_for_det:
                    methods_for_det.append(m)

        context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

        for method in methods_for_det:
            phi_runs_per_anomaly = {}
            for g_idx in top_k_global:
                g_idx = int(g_idx)
                runs = []
                for rep in range(reps):
                    res_kwargs = {
                        "seed": seed + rep,
                    }

                    if method in ("SHAP-CA", "LIME-CA"):
                        res_kwargs["context_weights"] = context_weights

                    res = explain_anomaly(
                        det,
                        det_name,
                        X=X_scaled,
                        idx=g_idx,
                        normal_idx=normal_idx_global,
                        method=method,
                        device=zone_id,
                        context_idx=context_idx,
                        **res_kwargs,
                    )
                    runs.append(res.phi)
                phi_runs_per_anomaly[g_idx] = runs

            # Chỉ tìm láng giềng đánh giá trong tập đã biết (Train)
            # Không dùng Test để tránh evaluation contamination
            eval_pool_indices = np.arange(0, train_end)

            bundle = compute_metrics(
                predict_fn, 
                X_scaled,
                top_k_global.tolist(),
                phi_runs_per_anomaly,
                eval_pool_idx=eval_pool_indices,
            )

            row = {
                "fold": fold.idx,
                "zone": zone_id,
                "detector": det_name,
                "method": method,
                "n_train": train_end,
                "n_test": test_end - val_end,
                "n_normal_pool": len(normal_idx_global),
                "theta": theta,
                "stability": bundle.stability,
                "fidelity": bundle.fidelity,
                "consistency": bundle.consistency,
                "sensitivity": bundle.sensitivity,
            }
            rows.append(row)

            if verbose:
                print(f"        [{method:9s}] "
                      f"stab={bundle.stability:.4f}  "
                      f"fid={bundle.fidelity:.6f}  "
                      f"cons={bundle.consistency:.4f}  "
                      f"sens={bundle.sensitivity:.4f}")

    return rows


# ============================================================
# Aggregate
# ============================================================

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std across folds cho mỗi (zone, detector, method)."""
    metric_cols = ["stability", "fidelity", "consistency", "sensitivity"]
    agg = df.groupby(["zone", "detector", "method"])[metric_cols].agg(
        ["mean", "std", "count"]
    )
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zones", nargs="+", default=ZONE_IDS,
                   choices=ZONE_IDS)
    p.add_argument("--detectors", nargs="+",
                   default=["IF", "LOF", "OCSVM", "MLP"])
    p.add_argument("--methods", nargs="+",
                   default=["SHAP-CA", "LIME-std", "LIME-CA"])
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/walk_forward_cubems"))
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 1 zone, 1 fold, IF only, reps=3.")
    p.add_argument(
        "--fold",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Chọn fold khi chạy quick mode (1, 2 hoặc 3)"
    )
    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.quick:
        if args.zones == ZONE_IDS:
            args.zones = ["floor2_z1"]

        # Fallback về ["IF"] chỉ khi người dùng không truyền cụ thể
        if "--detectors" not in sys.argv:
            args.detectors = ["IF"]

        if args.reps == 10:
            args.reps = 3

        selected_fold = args.fold if args.fold is not None else 1

        print(
            f"[QUICK MODE] zone={args.zones}, "
            f"detector={args.detectors}, "
            f"reps={args.reps}, "
            f"fold {selected_fold} only"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("WALK-FORWARD CUBEMS — 6 zones × 3 folds")
    print("=" * 72)
    print(f"Zones:      {args.zones}")
    print(f"Detectors:  {args.detectors}")
    print(f"Methods:    {args.methods}")
    print(f"Top-K:      {args.top_k}")
    print(f"Reps:       {args.reps}")
    print(f"Output:     {output_dir}")

    t_start = time.perf_counter()
    all_rows = []

    for zone_id in args.zones:
        print(f"\n{'#'*72}")
        print(f"# ZONE: {zone_id}")
        print(f"{'#'*72}")

        df = load_zone_dataframe(zone_id, verbose=False)
        X = df[FEATURE_COLUMNS].values.astype(np.float64)
        timestamps = df.index

        print(f"  Loaded: {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  Time: {timestamps[0].date()} -> {timestamps[-1].date()}")

        # Tạo Fold boundaries tự thích ứng theo độ dài dữ liệu thực tế
        folds_all = build_folds_for_length(len(X))

        if args.quick:
            selected_fold = args.fold if args.fold is not None else 1
            folds_all = [f for f in folds_all if f.idx == selected_fold]

        for fold in folds_all:
            print(f"\n  {'-'*68}")
            print(f"  Fold {fold.idx}")
            print(f"  {'-'*68}")
            print(f"    {fold.describe(timestamps)}")

            t0 = time.perf_counter()
            rows = run_fold(
                zone_id=zone_id, X=X, fold=fold,
                detector_names=args.detectors,
                methods=args.methods,
                top_k=args.top_k,
                reps=args.reps,
                seed=args.seed,
            )
            all_rows.extend(rows)
            print(f"\n    Fold {fold.idx} completed in "
                  f"{time.perf_counter()-t0:.1f}s")

    # Save
    df_all = pd.DataFrame(all_rows)
    per_fold_csv = output_dir / "per_fold_metrics.csv"
    df_all.to_csv(per_fold_csv, index=False)
    print(f"\n[Saved] Per-fold: {per_fold_csv}")

    df_agg = aggregate(df_all)
    agg_csv = output_dir / "aggregate_metrics.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f"[Saved] Aggregate: {agg_csv}")

    # Summary
    print(f"\n{'='*72}")
    print("FINAL SUMMARY — MEDIAN across (zone, detector)")
    print(f"{'='*72}")
    summary = df_all.groupby("method")[
        ["stability", "fidelity", "consistency", "sensitivity"]
    ].median()
    print(summary.to_string())

    total = time.perf_counter() - t_start
    print(f"\nTotal runtime: {total:.1f}s")


if __name__ == "__main__":
    main()