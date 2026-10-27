"""
inspect_domain_details.py
=========================
Chẩn đoán chi tiết cho 83 ca giao thoa ở Floor2 Z2 và Z4:
1. Thống kê xem feature nào đang đứng Top-1 nhiều nhất ở từng method.
2. Kiểm tra phân phối thứ hạng (Rank) của biến 'power'.
3. Xem giá trị thực (raw values) của các feature tại các điểm này.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_prep_cubems import (
    load_zone_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS, ZONE_IDS,
)
from anomaly_detection import build_detectors
from explainers import (
    get_predict_fn, compute_global_importance, compute_context_weights, explain_anomaly,
)
from walk_forward_cubems import build_folds_for_length

DETECTOR = "IF"
METHODS = ["LIME-std", "LIME-CA", "SHAP-CA"]
PERCENTILE = 95
LAMBDA_LUX = 100.0
LAMBDA_PLUG = 0.1
EPSILON_AC = 0.05
MIN_MINUTES = 30
SEED = 42

def load_groundtruth_rule_series(zone_id: str, target_hourly_index: pd.DatetimeIndex) -> pd.Series:
    floor = "Floor2" if "floor2" in zone_id else "Floor5"
    z_name = zone_id.split("_")[1]
    raw_path = ROOT_DIR / "data" / "raw" / f"2019{floor}.csv"
    if not raw_path.exists():
        raw_path = Path("data") / "raw" / f"2019{floor}.csv"

    df_raw = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
    ac_cols = [c for c in df_raw.columns if c.lower().startswith(f"{z_name}_ac")]
    plug_cols = [c for c in df_raw.columns if c.lower().startswith(f"{z_name}_plug")]
    lux_cols = [c for c in df_raw.columns if f"{z_name}_" in c.lower() and "lux" in c.lower()]

    p_ac_total = df_raw[ac_cols].fillna(0.0).sum(axis=1)
    p_plug = df_raw[plug_cols[0]].fillna(0.0)
    lux = df_raw[lux_cols[0]].ffill().bfill().fillna(0.0) if lux_cols else pd.Series(0.0, index=df_raw.index)

    unoccupied = (lux < LAMBDA_LUX) & (p_plug < LAMBDA_PLUG)
    ac_running = p_ac_total > EPSILON_AC
    hourly_violation = (unoccupied & ac_running).resample("1h").sum() >= MIN_MINUTES
    return hourly_violation.astype(int).reindex(target_hourly_index, fill_value=0)

def main():
    print("=" * 78)
    print("CHẨN ĐOÁN CHI TIẾT VECTOR GIẢI THÍCH CHO CÁC CA DỊ BIỆT GIAO THOA")
    print("=" * 78)

    detailed_records = []
    zones_to_check = ["floor2_z2", "floor2_z4"]
    power_idx = FEATURE_COLUMNS.index("power")
    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

    for zone_id in zones_to_check:
        df = load_zone_dataframe(zone_id, verbose=False)
        X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
        rule_series = load_groundtruth_rule_series(zone_id, df.index)
        rule_indices_all = set(np.where(rule_series.values == 1)[0])
        folds = build_folds_for_length(len(X_raw))

        for fold in folds:
            train_end, val_end, test_end = fold.train_end, fold.val_end, fold.test_end
            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(X_raw[:train_end])
            X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

            det = build_detectors(X_scaled[:train_end], [DETECTOR], random_state=SEED)[DETECTOR]
            predict_fn = get_predict_fn(det, DETECTOR)
            scores_all = det.score(X_scaled)
            theta = float(np.percentile(scores_all[train_end:val_end], PERCENTILE))

            test_indices = np.arange(val_end, test_end)
            det_flagged = set(test_indices[scores_all[val_end:test_end] > theta])
            rule_flagged = set([i for i in rule_indices_all if val_end <= i < test_end])
            concordance_set = sorted(list(det_flagged.intersection(rule_flagged)))

            if not concordance_set:
                continue

            theta_train = float(np.percentile(scores_all[:train_end], PERCENTILE))
            normal_idx = np.arange(0, train_end)[scores_all[:train_end] <= theta_train]
            global_importance = compute_global_importance(predict_fn, X_scaled[normal_idx], n_repeats=3, seed=SEED)
            context_weights = compute_context_weights(global_importance, FEATURE_COLUMNS, CONTEXT_COLUMNS)

            for g_idx in concordance_set[:15]:
                raw_row = df.iloc[g_idx]
                timestamp_str = str(df.index[g_idx])

                for method in METHODS:
                    extra_kwargs = {"device": zone_id, "context_idx": context_idx, "seed": SEED}
                    if "CA" in method:
                        extra_kwargs.update({"context_weights": context_weights, "frac": 0.40, "min_k": 50})

                    res = explain_anomaly(det, DETECTOR, X=X_scaled, idx=g_idx, normal_idx=normal_idx, method=method, **extra_kwargs)
                    abs_phi = np.abs(np.array(res.phi))
                    
                    # Tìm thứ hạng của các biến (từ lớn nhất đến nhỏ nhất)
                    ranked_indices = np.argsort(-abs_phi)
                    top1_feat = FEATURE_COLUMNS[ranked_indices[0]]
                    top2_feat = FEATURE_COLUMNS[ranked_indices[1]]
                    
                    # Thứ hạng của power (1-indexed: 1 là cao nhất, 7 là thấp nhất)
                    power_rank = int(np.where(ranked_indices == power_idx)[0][0]) + 1

                    detailed_records.append({
                        "zone": zone_id,
                        "fold": fold.idx,
                        "idx": g_idx,
                        "timestamp": timestamp_str,
                        "method": method,
                        "raw_power": raw_row["power"],
                        "raw_lux": raw_row["lux"],
                        "raw_temp": raw_row["temperature"],
                        "top1_feat": top1_feat,
                        "top2_feat": top2_feat,
                        "power_rank": power_rank,
                        "power_phi": res.phi[power_idx],
                    })

    df_details = pd.DataFrame(detailed_records)
    out_path = Path("results/domain_validation/inspection_details.csv")
    df_details.to_csv(out_path, index=False)
    print(f"\n[Đã lưu dữ liệu chi tiết tại: {out_path}]")

    print("\n" + "=" * 78)
    print("1. PHÂN PHỐI TẦN SUẤT ĐỨNG TOP-1 THEO TỪNG FEATURE (%):")
    print("=" * 78)
    top1_dist = pd.crosstab(df_details["method"], df_details["top1_feat"], normalize="index") * 100
    print(top1_dist.round(2).to_string())

    print("\n" + "=" * 78)
    print("2. PHÂN PHỐI TẦN SUẤT ĐỨNG TOP-2 THEO TỪNG FEATURE (%):")
    print("=" * 78)
    top2_dist = pd.crosstab(df_details["method"], df_details["top2_feat"], normalize="index") * 100
    print(top2_dist.round(2).to_string())

    print("\n" + "=" * 78)
    print("3. PHÂN PHỐI THỨ HẠNG CỦA BIẾN 'power' (1 = Top 1, 7 = Bét bảng):")
    print("=" * 78)
    rank_dist = pd.crosstab(df_details["method"], df_details["power_rank"], normalize="index") * 100
    print(rank_dist.round(2).to_string())
    print("\nThứ hạng trung bình của 'power' (Mean Rank):")
    print(df_details.groupby("method")["power_rank"].mean().round(2))

    print("\n" + "=" * 78)
    print("4. GIÁ TRỊ VẬT LÝ TRUNG BÌNH TẠI CÁC ĐIỂM DỊ BIỆT ĐƯỢC ĐÁNH GIÁ:")
    print("=" * 78)
    unique_cases = df_details[["zone", "idx", "raw_power", "raw_lux", "raw_temp"]].drop_duplicates()
    print(unique_cases.describe().round(2).to_string())

if __name__ == "__main__":
    main()