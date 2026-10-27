import pandas as pd
import numpy as np

# 1. Đọc file kết quả thô của walk-forward hoặc ablation
# (File được sinh ra từ walk_forward_cubems.py)
df = pd.read_csv("results/walk_forward_cubems/per_fold_metrics.csv")

# 2. Lọc riêng Isolation Forest và LIME-CA
df_if = df[(df["detector"] == "IF") & (df["method"] == "LIME-CA")]
print(f"Tổng số bản ghi đánh giá: {len(df_if)} (kỳ vọng: 18)")

# Cách A: Pooling toàn bộ 18 lượt chạy (Standard sample std across all 18 evaluations)
sens_all = df_if["sensitivity"].values
mean_pooled = np.mean(sens_all)
std_pooled_ddof1 = np.std(sens_all, ddof=1)
std_pooled_ddof0 = np.std(sens_all, ddof=0)
print(f"\n[CÁCH A - Pooled 18 Folds]:")
print(f"  Mean: {mean_pooled:.4f}")
print(f"  Std (ddof=1): {std_pooled_ddof1:.4f}")
print(f"  Std (ddof=0): {std_pooled_ddof0:.4f}")

# Cách B: Nhóm theo Zone trước rồi tính trung bình giữa các Zone
zone_means = df_if.groupby("zone")["sensitivity"].mean()
zone_stds = df_if.groupby("zone")["sensitivity"].std(ddof=1)
print(f"\n[CÁCH B - Grouped by 6 Zones]:")
print(f"  Std of 6 Zone Means: {zone_means.std(ddof=1):.4f}")
print(f"  Mean of 6 Zone Stds: {zone_stds.mean():.4f}")