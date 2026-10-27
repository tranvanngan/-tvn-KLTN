"""
plot_ablation.py
======================
Vẽ Boxplot chuẩn Times New Roman, đọc trực tiếp từ CSV,
loại bỏ hoàn toàn các chấm scatter/stripplot.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# 1. CẤU HÌNH FONT CHỮ TIMES NEW ROMAN CHUẨN TẠP CHÍ
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'stix',  # Giúp các ký hiệu [↑], [↓] đồng bộ font Times
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
})

# ============================================================
# 2. ĐỌC DỮ LIỆU TỪ CÁC FILE CSV
# ============================================================
path_per_fold = Path('results/walk_forward_cubems/per_fold_metrics.csv')
path_ablation = Path('results/ablation_context_selection.csv')

# Fallback nếu file nằm cùng thư mục hiện tại
if not path_per_fold.exists():
  path_per_fold = Path('per_fold_metrics.csv')
if not path_ablation.exists():
  path_ablation = Path('results_ablation_context_selection.csv')

df_main = pd.read_csv(path_per_fold)
df_lime_std = df_main[
    (df_main['method'] == 'LIME-std') & (df_main['detector'] == 'IF')
].copy()
df_lime_std['config'] = 'LIME-std'

df_abl = pd.read_csv(path_ablation)

cols = [
    'zone',
    'fold',
    'detector',
    'config',
    'consistency',
    'stability',
    'fidelity',
    'sensitivity',
]
df_plot = pd.concat([df_lime_std[cols], df_abl[cols]], ignore_index=True)

# Khóa thứ tự các cấu hình
config_order = ['LIME-std', 'Fixed-Cosine', 'Geom-Fixed', 'Geom-Learned']
df_plot['config'] = pd.Categorical(
    df_plot['config'], categories=config_order, ordered=True
)

# ============================================================
# 3. VẼ BIỂU ĐỒ BOXPLOT THUẦN TÚY (CLEAN BOXPLOT)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=300)
palette = ['#b0b0b0', '#4a7bb0', '#56b870', '#e05353']

metrics_info = [
    (
        'consistency',
        '(a) Explanation Consistency (Cosine) [↑]',
        'Cosine Similarity',
        axes[0],
    ),
    (
        'stability',
        '(b) Attribution Stability (CV) [↓]',
        'Coefficient of Variation',
        axes[1],
    ),
    (
        'fidelity',
        '(c) Local Surrogate Fidelity (MSE) [↓]',
        'Mean Squared Error',
        axes[2],
    ),
]

for col_name, title, ylabel, ax in metrics_info:
  sns.boxplot(
      data=df_plot,
      x='config',
      y=col_name,
      ax=ax,
      palette=palette,
      width=0.48,
      boxprops=dict(alpha=0.85, edgecolor='black', linewidth=1.1),
      medianprops=dict(color='black', linewidth=1.6),
      whiskerprops=dict(color='black', linewidth=1.1),
      capprops=dict(color='black', linewidth=1.1),
      flierprops=dict(
          marker='o',
          markersize=4,
          markerfacecolor='black',
          alpha=0.6,
          markeredgecolor='none',
      ),  # Chỉ hiển thị outlier thực sự
  )
  ax.set_title(title, fontweight='bold', pad=10)
  ax.set_ylabel(ylabel)
  ax.set_xlabel('')
  ax.tick_params(axis='x', rotation=12)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# Xuất ra thư mục figures
out_dir = Path('figures')
out_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(out_dir / 'figure3_ablation_clean.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'figure3_ablation_clean.png', bbox_inches='tight')
plt.show()

print(
    f'[Hoàn tất] Đã lưu hình vẽ Times New Roman tại: {out_dir / "figure3_ablation_clean.pdf"}'
)