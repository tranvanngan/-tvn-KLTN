"""
tune_fidelity_neighborhood.py

Setup:
- 6 zones x 3 walk-forward folds
- Base detector: Isolation Forest (seed = 42)
- Explainers: LIME-std, LIME-CA (Ours), SHAP-CA
- Metrics evaluated: Fidelity MSE at k in [25, 50, 100]
- Output: fidelity_neighborhood_sensitivity.csv & summary_table.csv
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
    load_zone_dataframe,
    FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
    ZONE_IDS,
)
from anomaly_detection import build_detectors
from explainers import (
    explain_anomaly,
    get_predict_fn,
    compute_global_importance,
    compute_context_weights,
)
from metrics import fidelity_mse
from walk_forward_cubems import build_folds_for_length

K_NEIGHBORS_LIST = [25, 50, 100]


def run_neighborhood_experiment(
    data_dir: Path,
    out_dir: Path,
    top_k: int = 20,
    seed: int = 42,
    beta_frac: float = 0.40,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_results = []

    print("=" * 80)
    print("FIDELITY NEIGHBORHOOD SENSITIVITY EXPERIMENT (k = 25, 50, 100)")
    print(f"Data directory: {data_dir}")
    print(f"Zones: {ZONE_IDS} | Detector: Isolation Forest | Top-K: {top_k}")
    print("=" * 80)

    for zone_id in ZONE_IDS:
        print(f"\n>>> Processing Zone: {zone_id}")
        csv_path = data_dir / f"{zone_id}_hourly.csv"
        
        # Gọi load_zone_dataframe tương thích
        try:
            df = load_zone_dataframe(zone_id, csv_file=csv_path, verbose=False)
        except TypeError:
            try:
                df = load_zone_dataframe(zone_id, path=csv_path, verbose=False)
            except TypeError:
                df = load_zone_dataframe(zone_id, verbose=False)

        X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)

        # Xây dựng 3 dynamic folds dựa trên độ dài dữ liệu thực tế
        folds_all = build_folds_for_length(len(X_raw))

        for fold in folds_all:
            train_end, val_end, test_end = fold.train_end, fold.val_end, fold.test_end
            train_indices = np.arange(0, train_end)

            # Causal Min-Max Scaling (chỉ fit trên train để tránh rò rỉ thông tin)
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(X_raw[:train_end])
            X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

            X_train = X_scaled[:train_end]
            X_val = X_scaled[train_end:val_end]
            X_test = X_scaled[val_end:test_end]

            # Fit Isolation Forest
            det_dict = build_detectors(X_train, detector_names=["IF"], random_state=seed)
            det = det_dict["IF"]
            predict_fn = get_predict_fn(det, "IF")

            scores_all = det.score(X_scaled)
            scores_train = scores_all[:train_end]
            scores_val = scores_all[train_end:val_end]
            scores_test = scores_all[val_end:test_end]

            # Calibration: Ngưỡng tính trên validation, pool bình thường lấy trên train
            theta = float(np.percentile(scores_val, 95))
            theta_train = float(np.percentile(scores_train, 95))
            normal_idx_global = train_indices[scores_train <= theta_train]

            # Tính trọng số ngữ cảnh cho CELF
            X_train_normal = X_scaled[normal_idx_global]
            global_importance = compute_global_importance(
                predict_fn, X_train_normal, n_repeats=5, seed=seed
            )
            context_weights = compute_context_weights(
                global_importance, FEATURE_COLUMNS, CONTEXT_COLUMNS
            )
            context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

            # Lấy mẫu dị biệt phân tầng trên tập test
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

            # Evaluation Pool: chỉ dùng tập train lịch sử
            eval_pool_indices = train_indices
            X_eval_pool = X_scaled[eval_pool_indices]

            methods = ["LIME-std", "LIME-CA", "SHAP-CA"]

            for method in methods:
                # 1. Thu thập phi vector cho từng điểm dị biệt
                phi_dict = {}
                res_kwargs = {"seed": seed}
                if method in ("SHAP-CA", "LIME-CA"):
                    res_kwargs["context_weights"] = context_weights

                for g_idx in top_k_global:
                    res = explain_anomaly(
                        det,
                        "IF",
                        X=X_scaled,
                        idx=int(g_idx),
                        normal_idx=normal_idx_global,
                        method=method,
                        device=zone_id,
                        context_idx=context_idx,
                        **res_kwargs,
                    )
                    phi_dict[g_idx] = res.phi

                # 2. Đánh giá Fidelity MSE qua từng k láng giềng
                for k_nei in K_NEIGHBORS_LIST:
                    fids = []
                    for g_idx in top_k_global:
                        x_target = X_scaled[g_idx]
                        phi = phi_dict[g_idx]
                        mse = fidelity_mse(
                            predict_fn,
                            X_eval_pool,
                            x_target,
                            phi,
                            n_neighbors=k_nei,
                        )
                        fids.append(mse)

                    mean_fid = float(np.mean(fids))
                    raw_results.append({
                        "zone": zone_id,
                        "fold": fold.idx,
                        "method": method,
                        "k_neighbors": k_nei,
                        "fidelity_mse": mean_fid,
                    })

            print(f"  [Fold {fold.idx}/3] Hoàn thành cho cả 3 methods ở các k: {K_NEIGHBORS_LIST}")

    # Xuất dữ liệu chi tiết
    df_raw = pd.DataFrame(raw_results)
    raw_csv_path = out_dir / "fidelity_neighborhood_sensitivity.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n[DONE] Đã lưu dữ liệu thô vào: {raw_csv_path}")

    # Tạo bảng tổng hợp số liệu
    generate_summary_table(df_raw, out_dir)


def generate_summary_table(df_raw: pd.DataFrame, out_dir: Path):
    records = []
    for k in K_NEIGHBORS_LIST:
        df_k = df_raw[df_raw["k_neighbors"] == k]
        
        # Tạo pivot table theo 18 lượt (zone x fold)
        pivot = df_k.pivot(index=["zone", "fold"], columns="method", values="fidelity_mse")
        
        lime_ca = pivot["LIME-CA"].values
        lime_std = pivot["LIME-std"].values
        shap_ca = pivot["SHAP-CA"].values

        # Relative difference trung bình theo cặp
        rel_diff_std = np.mean((lime_ca - lime_std) / lime_std) * 100.0
        rel_diff_shap = np.mean((lime_ca - shap_ca) / shap_ca) * 100.0

        for m in ["LIME-CA", "LIME-std", "SHAP-CA"]:
            vals = pivot[m].values
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            med_val = np.median(vals)
            q25 = np.percentile(vals, 25)
            q75 = np.percentile(vals, 75)

            records.append({
                "Neighborhood (k)": k,
                "Method": m,
                "Mean ± SD": f"{mean_val:.4f} ± {std_val:.4f}",
                "Median [IQR]": f"{med_val:.4f} [{q25:.4f}, {q75:.4f}]",
                "Relative Diff vs LIME-std (%)": f"{rel_diff_std:+.2f}%" if m == "LIME-CA" else "-",
                "Relative Diff vs SHAP-CA (%)": f"{rel_diff_shap:+.2f}%" if m == "LIME-CA" else "-",
            })

    df_summary = pd.DataFrame(records)
    summary_path = out_dir / "table_appendix_fidelity_neighborhood.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[DONE] Đã lưu bảng tổng hợp phụ lục vào: {summary_path}")
    print("\n" + df_summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/fidelity_neighborhood"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_neighborhood_experiment(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        top_k=args.top_k,
        seed=args.seed,
    )