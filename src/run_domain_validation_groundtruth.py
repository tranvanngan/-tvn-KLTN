"""
run_domain_validation_groundtruth.py
===================================
Thực nghiệm cho Mục 4.5: Domain-Informed Validation
- Trích xuất Ground-truth Rule R_AC trực tiếp từ cảm biến gốc (sub-metering: AC, Plug, Lux)
- Điều kiện: PAC > 0.05 kW, Lux < 100, Pplug < 0.1 kW và duy trì >= 30 phút trong mỗi giờ
- Đánh giá Detection Concordance (Recall_rule) trên Isolation Forest (ngưỡng 95%)
- Đánh giá Attribution Plausibility (rho_RC, Hit@1, 3 tiers) cho LIME-CA, SHAP-CA, LIME-std
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
LAMBDA_LUX = 100.0   # lux
LAMBDA_PLUG = 0.1     # kW
EPSILON_AC = 0.05     # kW
MIN_MINUTES = 30      # Số phút tối thiểu vi phạm trong 1 giờ để tính 1 giờ lỗi
SEED = 42

def load_groundtruth_rule_series(zone_id: str, target_hourly_index: pd.DatetimeIndex) -> pd.Series:
    floor = "Floor2" if "floor2" in zone_id else "Floor5"
    z_name = zone_id.split("_")[1]
    
    raw_path = ROOT_DIR / "data" / "raw" / f"2019{floor}.csv"
    if not raw_path.exists():
        raw_path = Path("data") / "raw" / f"2019{floor}.csv"
    if not raw_path.exists():
        raw_path = Path(f"2019{floor}.csv")

    df_raw = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
    
    ac_cols = [c for c in df_raw.columns if c.lower().startswith(f"{z_name}_ac")]
    plug_cols = [c for c in df_raw.columns if c.lower().startswith(f"{z_name}_plug")]
    lux_cols = [c for c in df_raw.columns if f"{z_name}_" in c.lower() and "lux" in c.lower()]

    if not ac_cols or not plug_cols:
        raise ValueError(f"Không tìm thấy cột AC hoặc Plug cho {zone_id} trong file {raw_path.name}")

    # Điền giá trị thiếu (forward fill rồi 0) để không làm mất mốc thời gian
    p_ac_total = df_raw[ac_cols].fillna(0.0).sum(axis=1)
    p_plug = df_raw[plug_cols[0]].fillna(0.0)

    # Nếu cột lux trong file gốc bị khuyết, nội suy hoặc dùng ngưỡng mặc định tắt đèn
    if lux_cols and not df_raw[lux_cols[0]].dropna().empty:
        lux = df_raw[lux_cols[0]].ffill().bfill().fillna(0.0)
    else:
        # Dự phòng: coi như phòng tắt đèn nếu ngoài giờ hành chính (sau 19h hoặc trước 7h)
        hours = df_raw.index.hour
        lux = pd.Series(np.where((hours >= 19) | (hours < 7), 0.0, 500.0), index=df_raw.index)

    unoccupied = (lux < LAMBDA_LUX) & (p_plug < LAMBDA_PLUG)
    ac_running = p_ac_total > EPSILON_AC
    minute_rule_violated = (unoccupied & ac_running).astype(int)

    hourly_violation_mins = minute_rule_violated.resample("1h").sum()
    hourly_rule_binary = (hourly_violation_mins >= MIN_MINUTES).astype(int)

    return hourly_rule_binary.reindex(target_hourly_index, fill_value=0)

def main():
    print("=" * 78)
    print("CHẠY THỰC NGHIỆM MỤC 4.5 TỪ DỮ LIỆU CẢM BIẾN GỐC (GROUND-TRUTH RULE)")
    print("=" * 78)

    power_idx = FEATURE_COLUMNS.index("power")
    lux_idx = FEATURE_COLUMNS.index("lux")
    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

    records_concordance = []
    records_attribution = []

    for zone_id in ZONE_IDS:
        print(f"\n>>> ZONE: {zone_id} <<<")
        df = load_zone_dataframe(zone_id, verbose=False)
        X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
        
        # 1. Tạo chuỗi nhãn ground-truth rule từ file raw 525k dòng
        rule_series = load_groundtruth_rule_series(zone_id, df.index)
        rule_indices_all = set(np.where(rule_series.values == 1)[0])
        print(f"Tổng số giờ vi phạm quy tắc R_AC (Ground-truth): {len(rule_indices_all)} / {len(df)} giờ")

        folds = build_folds_for_length(len(X_raw))

        for fold in folds:
            print(f"--- Fold {fold.idx} ---")
            train_end, val_end, test_end = fold.train_end, fold.val_end, fold.test_end

            scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            scaler.fit(X_raw[:train_end])
            X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

            # Huấn luyện Detector
            detectors = build_detectors(X_scaled[:train_end], [DETECTOR], random_state=SEED)
            det = detectors[DETECTOR]
            predict_fn = get_predict_fn(det, DETECTOR)

            scores_all = det.score(X_scaled)
            theta = float(np.percentile(scores_all[train_end:val_end], PERCENTILE))

            # Tập dị biệt trên Test
            scores_test = scores_all[val_end:test_end]
            test_indices = np.arange(val_end, test_end)
            det_flagged = set(test_indices[scores_test > theta])
            rule_flagged = set([i for i in rule_indices_all if val_end <= i < test_end])

            concordance_set = sorted(list(det_flagged.intersection(rule_flagged)))
            recall_rule = len(concordance_set) / len(rule_flagged) if len(rule_flagged) > 0 else np.nan

            records_concordance.append({
                "zone": zone_id,
                "fold": fold.idx,
                "n_rule": len(rule_flagged),
                "n_det": len(det_flagged),
                "n_concordant": len(concordance_set),
                "recall_rule": recall_rule,
            })

            if len(concordance_set) == 0:
                continue

            theta_train = float(np.percentile(scores_all[:train_end], PERCENTILE))
            normal_idx = np.arange(0, train_end)[scores_all[:train_end] <= theta_train]
            global_importance = compute_global_importance(predict_fn, X_scaled[normal_idx], n_repeats=3, seed=SEED)
            context_weights = compute_context_weights(global_importance, FEATURE_COLUMNS, CONTEXT_COLUMNS)

            # Lấy mẫu đại diện tối đa 15 ca giao thoa
            sample_eval = concordance_set[:15]

            for g_idx in sample_eval:
                for method in METHODS:
                    extra_kwargs = {
                        "device": zone_id,
                        "context_idx": context_idx,
                        "seed": SEED,
                    }
                    if "CA" in method:
                        extra_kwargs["context_weights"] = context_weights
                        extra_kwargs["frac"] = 0.40
                        extra_kwargs["min_k"] = 50

                    res = explain_anomaly(
                        det, DETECTOR, X=X_scaled, idx=g_idx,
                        normal_idx=normal_idx, method=method,
                        **extra_kwargs
                    )
                    phi = np.array(res.phi)
                    abs_phi = np.abs(phi)
                    total_mass = np.sum(abs_phi)

                    rho_rc = (abs_phi[power_idx] + abs_phi[lux_idx]) / total_mass if total_mass > 0 else 0.0
                    top1_hit = (np.argmax(abs_phi) == power_idx)

                    # Phân loại 3 tầng
                    if (rho_rc >= 0.50) and top1_hit:
                        tier = "Domain-Plausible"
                    elif (rho_rc >= 0.50) or top1_hit:
                        tier = "Partially-Plausible"
                    else:
                        tier = "Domain-Implausible"

                    records_attribution.append({
                        "zone": zone_id,
                        "fold": fold.idx,
                        "idx": g_idx,
                        "method": method,
                        "rho_rc": rho_rc,
                        "hit_at_1": int(top1_hit),
                        "tier": tier
                    })

    out_dir = Path("results/domain_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_conc = pd.DataFrame(records_concordance)
    df_conc.to_csv(out_dir / "detection_concordance_groundtruth.csv", index=False)

    df_attr = pd.DataFrame(records_attribution)
    df_attr.to_csv(out_dir / "attribution_plausibility_groundtruth.csv", index=False)

    # In kết quả
    print("\n" + "=" * 78)
    print("1. BÁO CÁO DETECTION CONCORDANCE (TỶ LỆ PHÁT HIỆN LỖI THEO GROUND-TRUTH RULE)")
    print("=" * 78)
    conc_summary = df_conc.groupby("zone")[["n_rule", "n_det", "n_concordant", "recall_rule"]].mean()
    conc_summary["recall_rule_%"] = conc_summary["recall_rule"] * 100
    print(conc_summary.round(2).to_string())
    print(f"\nTrung bình toàn bộ tòa nhà: Recall_rule = {df_conc['recall_rule'].mean()*100:.2f}%")

    print("\n" + "=" * 78)
    print("2. BÁO CÁO ATTRIBUTION PLAUSIBILITY (THEO PHƯƠNG PHÁP)")
    print("=" * 78)
    attr_summary = df_attr.groupby("method").agg(
        mean_rho_rc=("rho_rc", "mean"),
        hit_at_1_rate=("hit_at_1", "mean"),
        plausible_rate=("tier", lambda x: (x == "Domain-Plausible").mean()),
        partially_rate=("tier", lambda x: (x == "Partially-Plausible").mean()),
        implausible_rate=("tier", lambda x: (x == "Domain-Implausible").mean()),
    ) * 100
    print(attr_summary.round(2).to_string())
    print(f"\n[Hoàn tất] Kết quả đã được lưu tại: {out_dir}")

if __name__ == "__main__":
    main()