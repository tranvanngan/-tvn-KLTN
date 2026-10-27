"""plot_figure_domain_validation.py

================================
Đọc kết quả từ CSV và vẽ Figure 4 chuẩn IEEE/Elsevier với font Times New Roman.
- Panel (a): Top-1 Feature Distribution (Psychrometric Confounding)
- Panel (b): Electrical Load Diagnostic Sensitivity (Hit@1 vs Hit@Top-2)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Đường dẫn thư mục gốc Cubem
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INSPECT_CSV = ROOT_DIR / "results" / "domain_validation" / "inspection_details.csv"

if not INSPECT_CSV.exists():
  raise FileNotFoundError(
      f"Không tìm thấy file: {INSPECT_CSV}. Hãy chạy"
      " `python src/inspect_domain_details.py` trước."
  )

# 1. Đọc dữ liệu từ CSV
df_details = pd.read_csv(INSPECT_CSV)

methods = ["LIME-std", "LIME-CA", "SHAP-CA"]
methods_display = ["LIME-std", "LIME-CA (Ours)", "SHAP-CA"]

# 2. Tính toán Panel (a)
top1_counts = (
    pd.crosstab(
        df_details["method"], df_details["top1_feat"], normalize="index"
    )
    * 100
)

p_humidity = [
    top1_counts.loc[m, "humidity"] if "humidity" in top1_counts.columns else 0.0
    for m in methods
]
p_power = [
    top1_counts.loc[m, "power"] if "power" in top1_counts.columns else 0.0
    for m in methods
]
p_lux = [
    top1_counts.loc[m, "lux"] if "lux" in top1_counts.columns else 0.0
    for m in methods
]

other_cols = [
    c for c in top1_counts.columns if c not in ["humidity", "power", "lux"]
]
p_other = [top1_counts.loc[m, other_cols].sum() for m in methods]

# 3. Tính toán Panel (b)
hit_top1 = []
hit_top2 = []

for m in methods:
  sub = df_details[df_details["method"] == m]
  hit_top1.append((sub["power_rank"] == 1).mean() * 100)
  hit_top2.append((sub["power_rank"] <= 2).mean() * 100)

# ==========================================
# Cấu hình Font Times New Roman
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ------------------------------------------
# Panel (a): Top-1 Feature Distribution
# ------------------------------------------
x = np.arange(len(methods))
width = 0.52

ax1.bar(
    x,
    p_humidity,
    width,
    label="Humidity (Psychrometric symptom)",
    color="#4575b4",
)
ax1.bar(
    x,
    p_power,
    width,
    bottom=p_humidity,
    label="Power / $P_{\\mathrm{AC}}$ (Direct physical driver)",
    color="#d73027",
)
ax1.bar(
    x,
    p_lux,
    width,
    bottom=np.array(p_humidity) + np.array(p_power),
    label="Lux (Context proxy)",
    color="#fdae61",
)
ax1.bar(
    x,
    p_other,
    width,
    bottom=np.array(p_humidity) + np.array(p_power) + np.array(p_lux),
    label="Temporal / Temperature",
    color="#e0f3f8",
    edgecolor="#b0bec5",
)

ax1.axhline(
    100 / 7,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Random Baseline (14.29%)",
)
ax1.set_ylabel("Top-1 Attribution Share (%)")
ax1.set_title(
    "(a) Dominant Top-1 Feature Distribution\n(Psychrometric Confounding"
    " Mechanism)"
)
ax1.set_xticks(x)
ax1.set_xticklabels(methods_display, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.legend(loc="upper right", frameon=True)
ax1.grid(axis="y", linestyle=":", alpha=0.6)

for i in range(len(methods)):
  if p_power[i] > 0:
    ax1.text(
        x[i],
        p_humidity[i] + p_power[i] / 2,
        f"{p_power[i]:.1f}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10.5,
    )
  else:
    ax1.text(
        x[i],
        p_humidity[i] + 3,
        "0.0%",
        ha="center",
        va="bottom",
        color="#d73027",
        fontweight="bold",
        fontsize=10.5,
    )

# ------------------------------------------
# Panel (b): Hit@1 vs Hit@Top-2
# ------------------------------------------
x2 = np.arange(len(methods))
w2 = 0.35

r1 = ax2.bar(
    x2 - w2 / 2,
    hit_top1,
    w2,
    label="Hit@1 ($P_{\\mathrm{AC}}$ strictly Top-1)",
    color="#fc8d59",
)
r2 = ax2.bar(
    x2 + w2 / 2,
    hit_top2,
    w2,
    label="Hit@Top-2 ($P_{\\mathrm{AC}}$ in Top-2 Window)",
    color="#d73027",
)

ax2.axhline(
    100 / 7,
    color="gray",
    linestyle="--",
    linewidth=1.1,
    label="Random Hit@1 (14.29%)",
)
ax2.axhline(
    200 / 7,
    color="black",
    linestyle=":",
    linewidth=1.2,
    label="Random Hit@Top-2 (28.57%)",
)

ax2.set_ylabel("Attribution Hit Rate (%)")
ax2.set_title(
    "(b) Electrical Load Diagnostic Sensitivity\n(Single-feature vs. Top-2"
    " Operational Window)"
)
ax2.set_xticks(x2)
ax2.set_xticklabels(methods_display, fontweight="bold")
ax2.set_ylim(0, 55)
ax2.legend(loc="upper left", frameon=True)
ax2.grid(axis="y", linestyle=":", alpha=0.6)

for bar in r1:
  yval = bar.get_height()
  ax2.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 1,
      f"{yval:.1f}%",
      ha="center",
      va="bottom",
      fontsize=9.5,
  )

for bar in r2:
  yval = bar.get_height()
  ax2.text(
      bar.get_x() + bar.get_width() / 2,
      yval + 1,
      f"{yval:.1f}%",
      ha="center",
      va="bottom",
      fontweight="bold",
      fontsize=10,
  )

plt.tight_layout()

out_dir = ROOT_DIR / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "figure4_domain_validation_diagnostic.pdf"
png_path = out_dir / "figure4_domain_validation_diagnostic.png"

plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[Xong] File PDF đã lưu tại: {pdf_path}")
print(f"[Xong] File PNG đã lưu tại: {png_path}")