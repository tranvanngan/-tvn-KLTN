"""
src/run_case_study.py
Chạy phân tích đúng ca 2019-09-14 09:00 (Floor 2 Zone 4).
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from explainers import lime_std, lime_ca, shap_ca, get_predict_fn

print("=" * 78)
print("PHÂN TÍCH CA TIÊU BIỂU: FLOOR 2 ZONE 4 (2019-09-14 09:00)")
print("=" * 78)

# 1. Đọc dữ liệu
csv_path = Path("data/processed/floor2_z4_hourly.csv")
if not csv_path.exists():
    csv_path = list(Path(".").rglob("*floor2_z4*.csv"))[0]

df = pd.read_csv(csv_path)

# Đảm bảo thứ tự 7 cột đúng chuẩn:
# 0: power, 1: hour_sin, 2: hour_cos, 3: day_of_week, 4: temperature, 5: humidity, 6: lux
feature_cols = ["power", "hour_sin", "hour_cos", "day_of_week", "temperature", "humidity", "lux"]
matched_cols = []
for req in feature_cols:
    for c in df.columns:
        if req in c.lower() or (req == "day_of_week" and "day" in c.lower()):
            matched_cols.append(c)
            break
if len(matched_cols) != 7:
    matched_cols = df.select_dtypes(include=[np.number]).columns[:7].tolist()

X_raw = df[matched_cols].values.astype(float)
n_total = len(X_raw)

# Theo per_fold_metrics.csv đối với Fold 1:
tr_end = 3245   # Tập Train (Từ tháng 3/2019 đến đầu tháng 7/2019)
va_end = 4327   # Tập Val (Từ tháng 7/2019 đến giữa tháng 8/2019)
# target_idx = 4604 thuộc tập Test out-of-sample (4327 -> 5409)

scaler = MinMaxScaler()
scaler.fit(X_raw[:tr_end])
X = np.clip(scaler.transform(X_raw), 0.0, 1.0)

# Điểm bất thường mục tiêu
target_idx = 4604

# 3. Model Isolation Forest
X_train = X[:tr_end]
raw_if = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.05,
    bootstrap=False,
    random_state=42
)
raw_if.fit(X_train)
detector = SimpleNamespace(model=raw_if, name="IF")
predict_fn = get_predict_fn(detector, "IF")

# 4. Ngưỡng theta P95 từ Validation
X_val = X[tr_end:va_end]
val_scores = predict_fn(X_val)
theta = np.percentile(val_scores, 95)

# Tập bình thường D_normal từ Train
train_scores = predict_fn(X_train)
normal_idx = np.where(train_scores <= theta)[0]
context_idx = [1, 2, 3, 4, 5, 6]

# 5. Chạy giải thích
res_ca = lime_ca(
    detector, "IF", X, target_idx, normal_idx, context_idx,
    frac=0.40, min_k=50, M=150, sigma=0.1, kernel_width=0.75, ridge_alpha=1.0, seed=42
)

res_shap = shap_ca(
    detector, "IF", X, target_idx, normal_idx, context_idx,
    frac=0.40, min_k=50, M=64, seed=42
)

res_std = lime_std(
    detector, "IF", X, target_idx, normal_idx,
    M=150, sigma=0.1, kernel_width=0.75, ridge_alpha=1.0, seed=42
)

v_ca = np.asarray(res_ca.phi).flatten()
v_shap = np.asarray(res_shap.phi).flatten()
v_std = np.asarray(res_std.phi).flatten()

print("\n" + "=" * 78)
print(f"{'Thuộc tính':<16} | {'LIME-std':<14} | {'LIME-CA (Ours)':<16} | {'SHAP-CA':<14}")
print("-" * 78)
for j, name in enumerate(feature_cols):
    print(f"{name:<16} | {v_std[j]:+12.4f}   | {v_ca[j]:+14.4f}   | {v_shap[j]:+12.4f}")
print("=" * 78)

print("\nĐỐI CHIẾU VỚI TABLE E.1:")
print("-" * 78)
methods = [("LIME-CA", v_ca), ("SHAP-CA", v_shap), ("LIME-std", v_std)]
print(f"{'Phương pháp':<12} | {'Top-1':<12} | {'Top-2':<12} | {'Hạng Power':<12} | {'ϕ_power':<10}")
print("-" * 78)
for m_name, vals in methods:
    ranked_idx = np.argsort(np.abs(vals))[::-1]
    top1 = feature_cols[ranked_idx[0]]
    top2 = feature_cols[ranked_idx[1]]
    power_rank = int(np.where(ranked_idx == 0)[0][0]) + 1
    p_val = vals[0]
    print(f"{m_name:<12} | {top1:<12} | {top2:<12} | #{power_rank:<11} | {p_val:+8.4f}")
print("=" * 78)