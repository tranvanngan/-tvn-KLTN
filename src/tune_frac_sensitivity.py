"""
tune_frac_sensitivity.py
- Lấy mẫu bất thường theo ngưỡng phân vị theta = 95%
- Đánh giá trên đầy đủ các zone đại diện của tòa nhà.
- Khảo sát đồng thời cả 4 metrics: Consistency, Stability, Fidelity, Sensitivity.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep_cubems import load_zone_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS
from anomaly_detection import build_detectors
from explainers import (
    get_predict_fn,
    compute_global_importance,
    compute_context_weights,
    explain_anomaly,
)
from metrics import compute_metrics
from walk_forward_cubems import build_folds_for_length

# Cấu hình đồng bộ với pipeline chính
ZONES = ["floor2_z1", "floor2_z2", "floor2_z4", "floor5_z1", "floor5_z4", "floor5_z5"]
DETECTOR = "IF"
PERCENTILE_THETA = 95.0   # Ngưỡng phân vị bất thường 95% chuẩn
MAX_EVAL_ANOMALIES = 50   # Giới hạn lấy mẫu đều để kiểm soát thời gian chạy (set None nếu muốn chạy 100%)
REPS = 3
SEED = 42
FRAC_CANDIDATES = [0.15, 0.25, 0.35, 0.40, 0.50, 0.65]

def run_sensitivity():
    print("=" * 78)
    print("SENSITIVITY ANALYSIS CHO CONTEXT POOL FRACTION (PIPELINE CHUẨN ĐỒNG BỘ)")
    print("=" * 78)

    all_records = []

    for zone_id in ZONES:
        print(f"\n{'#'*78}\n# ZONE: {zone_id}\n{'#'*78}")
        df = load_zone_dataframe(zone_id, verbose=False)
        X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
        context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]
        
        folds = build_folds_for_length(len(X_raw))
        for fold in folds:
            print(f"\n--- Fold {fold.idx} ---")
            train_end, val_end, test_end = fold.train_end, fold.val_end, fold.test_end

            # 1. Walk-forward scaling chuẩn
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(X_raw[:train_end])
            X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

            # 2. Huấn luyện detector
            detectors = build_detectors(X_scaled[:train_end], [DETECTOR], random_state=SEED)
            det = detectors[DETECTOR]
            predict_fn = get_predict_fn(det, DETECTOR)

            # 3. Lọc normal pool và tính context weights
            scores_all = det.score(X_scaled)
            theta = float(np.percentile(scores_all[:train_end], PERCENTILE_THETA))
            normal_idx = np.arange(0, train_end)[scores_all[:train_end] <= theta]

            global_importance = compute_global_importance(
                predict_fn, X_scaled[normal_idx], n_repeats=3, seed=SEED
            )
            context_weights = compute_context_weights(
                global_importance, FEATURE_COLUMNS, CONTEXT_COLUMNS
            )

            # 4. Lọc bất thường trên tập test theo đúng ngưỡng theta = 95%
            scores_test = scores_all[val_end:test_end]
            test_indices_all = np.arange(val_end, test_end)
            anom_mask = scores_test > theta
            anom_global_indices = test_indices_all[anom_mask]

            if len(anom_global_indices) == 0:
                print(f"[Cảnh báo] Fold {fold.idx} không phát hiện bất thường vượt ngưỡng {PERCENTILE_THETA}%.")
                continue

            # Lấy mẫu ngẫu nhiên đồng đều nếu số lượng vượt quá MAX_EVAL_ANOMALIES
            rng = np.random.default_rng(SEED + fold.idx)
            if MAX_EVAL_ANOMALIES is not None and len(anom_global_indices) > MAX_EVAL_ANOMALIES:
                eval_anom_indices = rng.choice(anom_global_indices, size=MAX_EVAL_ANOMALIES, replace=False)
            else:
                eval_anom_indices = anom_global_indices

            print(f"Tổng số bất thường test (> theta): {len(anom_global_indices)} | Chọn đánh giá: {len(eval_anom_indices)}")

            for frac in FRAC_CANDIDATES:
                for method in ["SHAP-CA", "LIME-CA"]:
                    phi_runs = {}
                    for g_idx in eval_anom_indices:
                        g_idx = int(g_idx)
                        runs = []
                        for rep in range(REPS):
                            res = explain_anomaly(
                                det, DETECTOR, X=X_scaled, idx=g_idx,
                                normal_idx=normal_idx, method=method,
                                device=zone_id, context_idx=context_idx,
                                seed=SEED + rep, context_weights=context_weights,
                                frac=frac, min_k=50,
                            )
                            runs.append(res.phi)
                        phi_runs[g_idx] = runs

                    # Đánh giá metrics trên tập normal đã lọc sạch
                    bundle = compute_metrics(
                        predict_fn, X_scaled, eval_anom_indices.tolist(), phi_runs,
                        eval_pool_idx=normal_idx,
                    )
                    all_records.append({
                        "zone": zone_id,
                        "fold": fold.idx,
                        "frac": frac,
                        "method": method,
                        "consistency": bundle.consistency,
                        "stability": bundle.stability,
                        "fidelity": bundle.fidelity,
                        "sensitivity": bundle.sensitivity,
                    })

    res_df = pd.DataFrame(all_records)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frac_sensitivity_full_4metrics.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\n[Xong] Đã lưu dữ liệu chi tiết vào: {csv_path}")

    # ============================================================
    # TỔNG HỢP VÀ IN BẢNG CHUẨN HỌC THUẬT (MEAN & STD)
    # ============================================================
    print("\n" + "=" * 78)
    print("BẢNG TỔNG HỢP MEAN ± STD TOÀN BỘ CÁC ZONES VÀ FOLDS")
    print("=" * 78)

    summary = res_df.groupby(["method", "frac"])[
        ["consistency", "stability", "fidelity", "sensitivity"]
    ].agg(["mean", "std"]).round(4)
    print(summary)

if __name__ == "__main__":
    run_sensitivity()