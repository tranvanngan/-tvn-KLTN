from pathlib import Path
import pandas as pd
from scipy.stats import wilcoxon

# Trỏ thẳng vào thư mục kết quả mới chạy không chồng lấn
csv_path = Path("results/walk_forward_cubems_non_overlapping/per_fold_metrics.csv")
df = pd.read_csv(csv_path)

print("=" * 60)
print("KIỂM ĐỊNH WILCOXON MỚI (KHÔNG CHỒNG LẤN TUYỆT ĐỐI)")
print("=" * 60)

ca_data = df[df["method"] == "LIME-CA"].sort_values(["zone", "detector", "fold"])
std_data = df[df["method"] == "LIME-std"].sort_values(["zone", "detector", "fold"])

# Kiểm định trên toàn bộ 3 folds không chồng lấn mới
stat, p_val = wilcoxon(ca_data["consistency"], std_data["consistency"])
print(f"Giá trị p-value tổng hợp (3 Folds non-overlapping): p = {p_val:.6e}")

for f_idx in [1, 2, 3]:
    sub = df[df["fold"] == f_idx]
    ca_f = sub[sub["method"] == "LIME-CA"]["consistency"]
    std_f = sub[sub["method"] == "LIME-std"]["consistency"]
    stat_f, p_f = wilcoxon(ca_f, std_f)
    print(f"  - Riêng Fold {f_idx}: p = {p_f:.6e}")
print("=" * 60)