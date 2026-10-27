"""
plot_threshold_sensitivity.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# 1. CẤU HÌNH FONT TIMES NEW ROMAN CHUẨN TẠP CHÍ
# ------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'stix',  # Đồng bộ ký hiệu toán học theta, mu, mũi tên với font Times
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU TỰ ĐỘNG TỪ FILE CSV
# ------------------------------------------------------------
csv_path = Path("results/threshold_sensitivity/per_cell_metrics.csv")
if not csv_path.exists():
    csv_path = Path("per_cell_metrics.csv")

if not csv_path.exists():
    raise FileNotFoundError(f"[Lỗi] Không tìm thấy tệp dữ liệu 'per_cell_metrics.csv'. Hãy chạy thực nghiệm threshold_sensitivity.py trước.")

df = pd.read_csv(csv_path)

percentiles = [90, 92, 95, 97, 99]
mean_df = df.groupby(['percentile', 'method'])[['consistency', 'fidelity', 'stability']].mean().reset_index()

# ------------------------------------------------------------
# 3. THIẾT KẾ ĐỒ THỊ 3 PANEL (1 HÀNG x 3 CỘT)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), dpi=300)

styles = {
    'SHAP-CA': {'color': '#1f4e79', 'marker': 's', 'lw': 2.0, 'ms': 6.5, 'ls': '-'},
    'LIME-CA (Ours)': {'color': '#c00000', 'marker': 'o', 'lw': 2.2, 'ms': 7.0, 'ls': '-'},
    'LIME-std': {'color': '#7f7f7f', 'marker': '^', 'lw': 1.8, 'ms': 6.5, 'ls': '--'}
}

# (a) Explanation Consistency
ax1 = axes[0]
for m in ['SHAP-CA', 'LIME-CA (Ours)', 'LIME-std']:
    m_key = 'LIME-CA' if m == 'LIME-CA (Ours)' else m
    sub = mean_df[mean_df['method'] == m_key].sort_values('percentile')
    ax1.plot(sub['percentile'], sub['consistency'], label=m, **styles[m])

ax1.axvline(x=95, color='#d9534f', linestyle=':', linewidth=1.2, alpha=0.7)
ax1.set_title(r'(a) Explanation Consistency [$\uparrow$]', fontweight='bold', pad=10)
ax1.set_xlabel(r'Anomaly Threshold Percentile $\theta$ (%)')
ax1.set_ylabel('Cosine Similarity')
ax1.set_xticks(percentiles)
ax1.set_ylim(0.35, 0.85)
ax1.grid(True, linestyle='--', alpha=0.45)
ax1.legend(loc='lower right', frameon=True, framealpha=0.9)

# (b) Local Surrogate Fidelity (MSE)
ax2 = axes[1]
for m in ['SHAP-CA', 'LIME-std', 'LIME-CA (Ours)']:
    m_key = 'LIME-CA' if m == 'LIME-CA (Ours)' else m
    sub = mean_df[mean_df['method'] == m_key].sort_values('percentile')
    ax2.plot(sub['percentile'], sub['fidelity'], label=m, **styles[m])

ax2.axvline(x=95, color='#d9534f', linestyle=':', linewidth=1.2, alpha=0.7)
ax2.set_title(r'(b) Local Surrogate Fidelity (MSE) [$\downarrow$]', fontweight='bold', pad=10)
ax2.set_xlabel(r'Anomaly Threshold Percentile $\theta$ (%)')
ax2.set_ylabel('Mean Squared Error')
ax2.set_xticks(percentiles)
ax2.set_ylim(0.00, 0.55)
ax2.grid(True, linestyle='--', alpha=0.45)
ax2.legend(loc='upper left', frameon=True, framealpha=0.9)

# Chú thích điểm chuẩn 95% an toàn
try:
    benchmark_fidelity = mean_df[(mean_df['method']=='LIME-CA') & (mean_df['percentile']==95)]['fidelity'].values[0]
    ax2.annotate(r'Benchmark Operating Point ($\theta=95\%$)',
                 xy=(95, benchmark_fidelity),
                 xytext=(93, 0.22),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                 ha='center', fontsize=9.5,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#cccc99', alpha=0.9))
except Exception:
    pass

# (c) Attribution Stability (CV)
ax3 = axes[2]
for m in ['LIME-std', 'LIME-CA (Ours)', 'SHAP-CA']:
    m_key = 'LIME-CA' if m == 'LIME-CA (Ours)' else m
    sub = mean_df[mean_df['method'] == m_key].sort_values('percentile')
    ax3.plot(sub['percentile'], sub['stability'], label=m, **styles[m])

ax3.axvline(x=95, color='#d9534f', linestyle=':', linewidth=1.2, alpha=0.7)
ax3.set_title(r'(c) Attribution Stability (CV) [$\downarrow$]', fontweight='bold', pad=10)
ax3.set_xlabel(r'Anomaly Threshold Percentile $\theta$ (%)')
ax3.set_ylabel('Coefficient of Variation')
ax3.set_xticks(percentiles)
ax3.set_ylim(0.12, 0.36)
ax3.grid(True, linestyle='--', alpha=0.45)
ax3.legend(loc='lower left', frameon=True, framealpha=0.9)

plt.tight_layout()

# Xuất file ảnh
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "figure2_threshold_sensitivity.pdf", bbox_inches='tight')
plt.savefig(out_dir / "figure2_threshold_sensitivity.png", bbox_inches='tight')
plt.show()

print(f"[Xong] Đã vẽ và lưu biểu đồ thành công tại: {out_dir / 'figure2_threshold_sensitivity.pdf'}")