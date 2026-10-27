import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# CẤU HÌNH GLOBAL FONT: TIMES NEW ROMAN
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# HÌNH 1: SỰ CỐ VẬN HÀNH 24 GIỜ (14/09/2019 - Floor 2 Zone 4)
# ==============================================================================

hours = np.arange(24)

power = np.array([
    0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03,
    8.132, 6.642, 4.384,
    0.03, 0.03, 0.042, 0.21, 0.052, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03
])

temp = np.array([
    24.06, 24.09, 24.38, 24.79, 24.42, 24.12, 24.18, 24.23,
    19.93, 18.34, 18.67,
    21.99, 23.16, 23.56, 23.81, 24.20, 24.22, 24.22, 24.20, 24.18, 24.23, 24.30, 24.34, 24.37
])

humidity = np.array([
    75.24, 75.36, 75.26, 75.27, 75.51, 75.84, 75.43, 75.21,
    73.56, 79.34, 80.49,
    76.53, 75.09, 74.65, 74.79, 75.50, 76.47, 77.29, 77.22, 76.12, 75.32, 75.74, 75.90, 75.92
])

fig, ax1 = plt.subplots(figsize=(10, 4.8), dpi=300)

# Trục Trái: Active Power (kW)
color1 = '#d9534f'
ax1.set_xlabel('Hour of Day (2019-09-14)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Active Power (kW)', color=color1, fontsize=11, fontweight='bold')
line1 = ax1.plot(hours, power, color=color1, marker='o', linewidth=2.5, label='Active Power (kW)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-0.5, 10.5)
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, linestyle=':', alpha=0.6)

# Trục Phải: Nhiệt độ (°C) và Độ ẩm (%) chung một trục
ax2 = ax1.twinx()
color2 = '#0275d8'
line2 = ax2.plot(hours, temp, color=color2, linestyle='--', marker='s', linewidth=2, label='Temperature (°C)')

color3 = '#2ca02c'
line3 = ax2.plot(hours, humidity, color=color3, linestyle='-.', marker='^', linewidth=2, label='Relative Humidity (%)')

ax2.set_ylabel('Temperature (°C) / Humidity (%)', color='#333333', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y')
ax2.set_ylim(15, 95)  # Mở rộng dải một chút để hiển thị số không bị sát mép trên

# Các mốc giờ cần hiển thị số liệu
target_hours = [8, 9, 10]

# Hiển thị số liệu cho Active Power (tại ax1)
for h in target_hours:
    val = power[h]
    ax1.annotate(f'{val:.3f}', xy=(h, val), xytext=(0, 8), textcoords='offset points',
                 ha='center', fontsize=8, fontweight='bold', color=color1)

# Hiển thị số liệu cho Temperature (tại ax2)
for h in target_hours:
    val = temp[h]
    ax2.annotate(f'{val:.2f}', xy=(h, val), xytext=(-12, -15), textcoords='offset points',
                 ha='center', fontsize=8, fontweight='bold', color=color2)

# Hiển thị số liệu cho Relative Humidity (tại ax2)
for h in target_hours:
    val = humidity[h]
    ax2.annotate(f'{val:.2f}', xy=(h, val), xytext=(15, 6), textcoords='offset points',
                 ha='center', fontsize=8, fontweight='bold', color=color3)

# Cắm cờ đánh dấu mốc sự cố 09:00
ax1.axvline(x=9, color='darkorange', linestyle='--', linewidth=2.5)

# Chú thích Legend
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines] + ['Anomaly Flagged (09:00)']
flag_line = plt.Line2D([0], [0], color='darkorange', linestyle='--', linewidth=2.5)
ax1.legend(lines + [flag_line], labels, loc='upper right', frameon=True, fontsize=9)

plt.title('Weekend Cooling Override Event (Floor 2 Zone 4 - 2019-09-14)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figure_24h_telemetry.png')
plt.close()


# ==============================================================================
# HÌNH 2: BIỂU ĐỒ THANH SO SÁNH TRỌNG SỐ GIẢI THÍCH
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharex=False)

data_std = {
    'Humidity': 0.0420, 'Day of Week': 0.0411, 'Lux': 0.0258, 
    'Hour (Sin)': 0.0083, 'Hour (Cos)': 0.0059, 'Active Power': -0.0009, 'Temperature': -0.0266
}
data_ca = {
    'Active Power': 0.0866, 'Hour (Sin)': 0.0112, 'Day of Week': 0.0041, 
    'Humidity': -0.0010, 'Temperature': -0.0464, 'Hour (Cos)': -0.0670, 'Lux': -0.0846
}
data_shap = {
    'Active Power': 0.0638, 'Temperature': 0.0391, 'Hour (Cos)': 0.0051, 
    'Lux': 0.0033, 'Humidity': -0.0022, 'Day of Week': -0.0033, 'Hour (Sin)': -0.0033
}

methods = [('(a) LIME-std Baseline', data_std), ('(b) LIME-CA (Ours)', data_ca), ('(c) SHAP-CA (Ours)', data_shap)]

for ax, (title, d) in zip(axes, methods):
    features = list(d.keys())[::-1]
    values = list(d.values())[::-1]
    colors = ['#d9534f' if f == 'Active Power' else '#5bc0de' if v > 0 else '#aaaaaa' for f, v in zip(features, values)]
    
    ax.barh(features, values, color=colors, edgecolor='black', alpha=0.85, height=0.6)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Attribution Value ($\phi$)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('figure_local_attributions.png')
plt.close()

print("--> Đã hiển thị số liệu của cả 3 đường tại mốc 8h, 9h, 10h thành công!")