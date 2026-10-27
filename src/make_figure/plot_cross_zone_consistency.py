"""
plot_cross_zone_heatmaps.py
===========================
Tạo Heatmap 4-panel cho Cross-zone Analysis tự động đọc từ tệp kết quả thực nghiệm.
Chuẩn font Times New Roman, hiển thị rõ ràng 6 Zones x 3 Phương pháp.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Cấu hình phông chữ Times New Roman
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10.5,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10.5,
})

# 2. Tự động đọc dữ liệu từ thư mục kết quả non-overlapping
results_path = (
    Path('results/walk_forward_cubems_non_overlapping/per_fold_metrics.csv')
)
if not results_path.exists():
  raise FileNotFoundError(
      f'[Lỗi] Không tìm thấy tệp kết quả tại đường dẫn: {results_path}. Hãy'
      ' đảm bảo bạn đã chạy thực nghiệm trước.'
)

df_raw = pd.read_csv(results_path)

# Lọc các phương pháp cần thiết và đổi tên 'LIME-CA' thành 'LIME-CA (Ours)' cho khớp với bài báo
methods_map = {
    'LIME-std': 'LIME-std',
    'LIME-CA': 'LIME-CA (Ours)',
    'SHAP-CA': 'SHAP-CA',
}
df_filtered = df_raw[df_raw['method'].isin(methods_map.keys())].copy()
df_filtered['method'] = df_filtered['method'].map(methods_map)

# Tính trung bình (mean) theo từng Zone và Method (gộp qua các folds và detectors)
df_agg = (
    df_filtered.groupby(['zone', 'method'])[
        ['consistency', 'stability', 'fidelity', 'sensitivity']
    ]
    .mean()
    .reset_index()
)

# Chuẩn hóa tên Zone nếu cần (ví dụ: 'floor2_z1' -> 'Floor2 Z1')
zone_map = {
    'floor2_z1': 'Floor2 Z1',
    'floor2_z2': 'Floor2 Z2',
    'floor2_z4': 'Floor2 Z4',
    'floor5_z1': 'Floor5 Z1',
    'floor5_z4': 'Floor5 Z4',
    'floor5_z5': 'Floor5 Z5',
}
df_agg['zone'] = df_agg['zone'].map(zone_map)

# 3. Khởi tạo 4 panel Heatmap
fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), dpi=300)

metrics_meta = [
    ('consistency', '(a) Consistency [↑]', 'YlGnBu'),
    ('stability', '(b) Stability (CV) [↓]', 'YlOrRd_r'),
    ('fidelity', '(c) Fidelity (MSE) [↓]', 'Blues_r'),
    ('sensitivity', '(d) Sensitivity [↑]', 'Purples'),
]

methods_order = ['LIME-std', 'LIME-CA (Ours)', 'SHAP-CA']
zones_order = [
    'Floor2 Z1',
    'Floor2 Z2',
    'Floor2 Z4',
    'Floor5 Z1',
    'Floor5 Z4',
    'Floor5 Z5',
]

for idx, (met, title, cmap) in enumerate(metrics_meta):
  pivot = df_agg.pivot(index='zone', columns='method', values=met)
  # Sắp xếp lại thứ tự hàng và cột theo chuẩn bài báo
  pivot = pivot.reindex(index=zones_order, columns=methods_order)

  ax = axes[idx]
  sns.heatmap(
      pivot,
      annot=True,
      fmt='.3f',
      cmap=cmap,
      cbar=True,
      ax=ax,
      linewidths=0.6,
      linecolor='white',
      annot_kws={'size': 9.5},
  )
  ax.set_title(title, fontweight='bold', pad=10)
  ax.set_xlabel('')
  if idx > 0:
    ax.set_ylabel('')
    ax.set_yticks([])
  else:
    ax.set_ylabel('Building Zones', fontweight='bold')
  ax.tick_params(axis='x', rotation=25)

plt.tight_layout()

# Xuất file đầu ra
out_dir = Path('figures')
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / 'figure2_cross_zone_heatmaps.pdf', bbox_inches='tight')
plt.savefig(out_dir / 'figure2_cross_zone_heatmaps.png', bbox_inches='tight')
plt.close()

print(
    '[Hoàn tất] Đã tự động cập nhật và lưu Heatmap 4-panel từ dữ liệu mới nhất'
    f' tại: {out_dir / "figure2_cross_zone_heatmaps.pdf"}'
)