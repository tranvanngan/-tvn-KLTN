import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Cấu hình style chuẩn tạp chí khoa học (IEEE / Elsevier)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10.5,
    'figure.titlesize': 13
})

# 2. Đọc file dữ liệu thực nghiệm 216 runs
csv_path = Path("results/frac_sensitivity_full_4metrics.csv")
if not csv_path.exists():
    csv_path = Path("frac_sensitivity_full_4metrics.csv")

df = pd.read_csv(csv_path)

# Tính Mean và Std theo từng phương pháp và mức frac
summary = df.groupby(['method', 'frac'])[['consistency', 'stability', 'fidelity', 'sensitivity']].agg(['mean', 'std']).reset_index()

# 3. Khởi tạo Figure 2x2
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)

metrics_config = [
    ('consistency', 'Attribution Consistency (↑)', axes[0, 0]),
    ('stability', 'Instability / Variance (↓)', axes[0, 1]),
    ('fidelity', 'Fidelity Error (↓)', axes[1, 0]),
    ('sensitivity', 'Feature Sensitivity (↑)', axes[1, 1])
]

methods = ['LIME-CA', 'SHAP-CA']
styles = {
    'LIME-CA': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
    'SHAP-CA': {'color': '#d62728', 'marker': 's', 'linestyle': '-'}
}

# 4. Vẽ từng đồ thị con
for col_name, title_name, ax in metrics_config:
    for method in methods:
        sub = summary[summary['method'] == method]
        x = sub['frac']
        y = sub[(col_name, 'mean')]
        yerr = sub[(col_name, 'std')]
        
        ax.errorbar(
            x, y, yerr=yerr, label=method,
            color=styles[method]['color'],
            marker=styles[method]['marker'],
            linestyle=styles[method]['linestyle'],
            capsize=3.5, capthick=1.2, elinewidth=1.2,
            linewidth=1.8, markersize=6
        )
    
    # Kẻ đường gióng dọc tại điểm chọn frac = 0.40
    ax.axvline(x=0.40, color='gray', linestyle='--', linewidth=1.2, alpha=0.85)
    
    ax.set_ylabel(title_name)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Chú thích mũi tên tại điểm chọn ở panel Fidelity
    if col_name == 'fidelity':
        ax.annotate(
            'Selected point\n(frac = 0.40)', 
            xy=(0.40, 0.1088), xytext=(0.43, 0.25),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
            fontsize=9.5, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
        )

# Thiết lập nhãn trục X và các mốc giá trị thực tế
axes[1, 0].set_xlabel(r'Background Fraction ($\beta_{\mathrm{frac}}$)')
axes[1, 1].set_xlabel(r'Background Fraction ($\beta_{\mathrm{frac}}$)')

frac_ticks = [0.15, 0.25, 0.35, 0.40, 0.50, 0.65]
for ax in axes.flat:
    ax.set_xticks(frac_ticks)
    ax.set_xticklabels([str(t) for t in frac_ticks])

# Legend chung nằm ở phía trên đồ thị
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Lưu file ảnh độ phân giải cao
output_img = "figure_frac_sensitivity.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.savefig("figure_frac_sensitivity.pdf", bbox_inches='tight') # Xuất thêm file PDF vector cho LaTeX
print(f"Đã lưu biểu đồ vào {output_img} và figure_frac_sensitivity.pdf")