import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# 1. Đọc tệp dữ liệu per_fold_metrics.csv
# (Nếu chạy ngoài thư mục dự án, bạn sửa đường dẫn file cho đúng)
file_path = "results/walk_forward_cubems/per_fold_metrics.csv" 

try:
    df = pd.read_csv(file_path)
    print("✅ Đã tải dữ liệu thành công!\n")
except FileNotFoundError:
    print(f"❌ Không tìm thấy tệp '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

methods = ['SHAP-CA', 'LIME-std', 'LIME-CA', 'DeepSHAP', 'IG']
metrics = ['stability', 'fidelity', 'consistency', 'sensitivity']

print("=" * 85)
print("📌 KẾT QUẢ THỐNG KÊ ĐỐI SOÁT BẢNG 3 (TABLE 3 VERIFICATION)")
print("=" * 85)

# 2. In thống kê chi tiết từng Metric cho từng Phương pháp
for metric in metrics:
    direction = "(↑ Càng lớn càng tốt)" if metric == 'consistency' else "(↓ Càng nhỏ càng tốt)"
    print(f"\n📊 METRIC: {metric.upper()} {direction}")
    print(f"{'Method':<12} | {'Mean ± Std':<22} | {'Median [Q25, Q75]':<28} | N")
    print("-" * 75)
    
    for m in methods:
        sub = df[df['method'] == m][metric].dropna()
        if len(sub) == 0:
            continue
            
        mean_val = sub.mean()
        std_val  = sub.std(ddof=1)
        med_val  = sub.median()
        q25_val  = sub.quantile(0.25)
        q75_val  = sub.quantile(0.75)
        
        print(f"{m:<12} | {mean_val:.4f} ± {std_val:.4f}     | {med_val:.4f} [{q25_val:.4f}, {q75_val:.4f}] | {len(sub)}")

# 3. So sánh ghép cặp LIME-CA vs LIME-std (N = 72 evaluations)
print("\n" + "=" * 85)
print("📌 TÍNH TỶ LỆ THẮNG (WIN-RATE) & WILCOXON P-VALUE (LIME-CA vs LIME-std)")
print("=" * 85)

lime_ca_df  = df[df['method'] == 'LIME-CA'].set_index(['zone', 'detector', 'fold'])
lime_std_df = df[df['method'] == 'LIME-std'].set_index(['zone', 'detector', 'fold'])

merged = lime_ca_df.join(lime_std_df, lsuffix='_ca', rsuffix='_std')

for metric in metrics:
    ca_vals  = merged[f'{metric}_ca']
    std_vals = merged[f'{metric}_std']
    
    if metric in ['consistency', 'sensitivity']: 
        wins = (ca_vals > std_vals).sum()
    else:
        # Stability, Fidelity càng thấp càng tốt
        wins = (ca_vals < std_vals).sum()
        
    win_rate = (wins / len(merged)) * 100
    stat, p_val = wilcoxon(ca_vals, std_vals)
    
    print(f"• {metric.capitalize():<12}: LIME-CA tốt hơn {wins}/{len(merged)} lần ({win_rate:.1f}%) | Wilcoxon p-value = {p_val:.6e}")