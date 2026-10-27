# CUBEMS — Working Notes cho Paper CLEF v2

> Ghi chú ngắn gọn các quyết định methodology và insight thu được trong
> quá trình xây dựng pipeline trên dataset CUBEMS. Dùng làm reference khi
> viết paper, đặc biệt cho Section 3 (Methodology) và Section 4 (Setup).

---

## 📌 Dataset Overview

**CUBEMS** = Chulalongkorn University Building Energy Management System.

| Thuộc tính | Giá trị |
|-----------|---------|
| Tòa nhà | 7 tầng (M5 building) |
| Thời gian | 18 tháng (Jul 2018 – Dec 2019) |
| Sampling | 1 phút/lần |
| Files sử dụng | `2019Floor2.csv`, `2019Floor5.csv` (chỉ dùng năm 2019) |
| Cấu trúc cột | `<zone>_<device>(kW)`, `<zone>_S1(degC/RH%/lux)` |
| Số samples sau resample | 8,760 giờ/năm |

**Lý do chọn năm 2019 (bỏ 2018):**
- File 2019 có 12 tháng liên tục → đủ mùa, đủ seasonal variation
- File 2018 chỉ 6 tháng (Jul–Dec) → không đủ temporal coverage

**Lý do chọn Floor 2 và Floor 5:**
- Floor 2: đa dạng cấu hình zone (z1 có 1 AC, z2 có tới 14 AC, z3 không AC) → stress-test framework
- Floor 5: cấu trúc tương tự Floor 3, 4, 7 (5 zones, mỗi zone có 1 AC) → tính đại diện cao

---

## 📊 Data Quality Findings

### Missing values (trước resample, ở granularity 1 phút)

| Loại | Missing range |
|------|:---:|
| Power columns (`*_AC*`, `*_Light`, `*_Plug`) | 0.1% – 5.7% |
| Sensor columns (`*_S1(degC)`, `*_S1(RH%)`, `*_S1(lux)`) | 18% – 43% |

**Insight:** Sensors mất nhiều hơn power đáng kể. Lý do có thể: sensor gửi dữ liệu
không đều, hoặc hệ thống ghi log chỉ ghi khi có thay đổi.

### Tác động của resampling lên missing

Ở 1 phút: 20% missing. Sau khi aggregate lên 1 giờ (mean 60 samples):
- Nếu missing là random → chỉ còn **~1–3% missing** ở hourly
- Có thể drop hàng bị missing hoặc interpolate

→ **Sensor features vẫn usable sau resample** dù trông "xấu" ở 1 phút.

---

## 🎯 Zone Selection

Ban đầu Floor 2 có 4 zones, Floor 5 có 5 zones = 9 zones.

**Bỏ 3 zones với lý do:**

| Zone | Lý do bỏ |
|------|----------|
| floor2_z3 | Chỉ có Light + Plug, không có AC → power mean 0.83 kW → **không có anomaly signal** |
| floor5_z3 | Tương tự floor2_z3 (mean 0.63 kW) |
| floor5_z2 | Sensors missing 42.57% → **quá nhiều missing** ngay cả sau resample |

**→ Còn lại 6 zones:**

| Zone ID | Floor | Số ACs | Power mean (kW) | Đặc điểm |
|---------|:---:|:---:|:---:|---------|
| floor2_z1 | 2 | 1 | 11.79 | Standard |
| **floor2_z2** | 2 | **14** | 12.26 | **Đặc biệt — multi-AC** |
| floor2_z4 | 2 | 1 | 2.63 | Low-power |
| floor5_z1 | 5 | 4 | 7.03 | Multi-AC |
| floor5_z4 | 5 | 1 | 5.71 | Standard |
| floor5_z5 | 5 | 1 | 2.84 | Low-power |

→ **6 devices** với 3 loại operational pattern:
- Single AC (z1, z4, z5)
- Multi AC (floor2_z2, floor5_z1)
- No AC / low-power (đã bỏ)

---

## 🔧 Feature Engineering

### Power aggregation

Mỗi zone gộp tất cả cột `(kW)` thành **1 cột `power`** duy nhất:
power = Σ(AC_power) + Light_power + Plug_power

text

**Lý do:**
- Building manager quan tâm **zone-level consumption** hơn từng thiết bị riêng lẻ
- Anomaly signal rõ hơn khi gộp (noise averaging)
- Cần 1 target variable cho CLEF framework

### Context features

**Đề xuất 3 context features:**

| Feature | Nguồn | Vai trò |
|---------|-------|---------|
| `hour` | Derived từ timestamp | Temporal context |
| `temperature` | `*_S1(degC)` | Environmental context |
| `humidity` | `*_S1(RH%)` | Environmental context |

**Cân nhắc `lux`:**
- `lux` có thể làm context tốt (ánh sáng ↔ hoạt động)
- Nhưng lux có thể khác nhau giữa zones
- Sẽ quyết định sau khi test

**Khác biệt với dataset cũ:** CUBEMS **không có occupancy sensor** công khai
→ Sẽ bỏ "Alignment Score" (human vs device) hoặc thay bằng temporal attribution.

### Feature vector cuối cùng
Feature vector cho mỗi zone:
[power, hour, day_of_week, temperature, humidity, lux]

Context vector:
[hour, temperature, humidity] (hoặc thêm lux)

text

**Cấu trúc giống hệt CLEF gốc** → pipeline anomaly detection + explainers
có thể chạy lại không cần thay đổi code.

---

## 🔬 Methodology Decisions

### Walk-forward cross-validation

**Lý do dùng walk-forward thay vì single split:**

1. **Boundary effect:** Single split đặt test ở cuối dataset → samples cuối
   bị detector flag là anomaly một cách hệ thống (vì không có future context
   để compare). Walk-forward với test windows không overlap giữa các folds
   sẽ trung bình hóa effect này.

2. **Statistical power:** Single split chỉ có 1 test set. Walk-forward với
   3 folds cho phép báo cáo mean ± std → robust hơn.

3. **Standard practice:** Đây là chuẩn của các benchmark time-series anomaly
   detection (NAB, NASA SMAP/MSL, Yahoo S5).

**Cấu trúc 3 folds (expanding window):**
Fold 1: Train=[0:145] Val=[145:181] Test=[181:241]
Fold 2: Train=[0:193] Val=[193:241] Test=[241:301]
Fold 3: Train=[0:241] Val=[241:301] Test=[301:355]

text

### Anomaly detection protocol
Fit detector CHỈ trên train slice

Score toàn bộ X (dùng detector đã fit)

theta = 95th percentile của train scores

Normal pool = train indices có score <= theta

Test anomalies = top-K samples trong test slice theo anomaly score

Explain: normal pool từ train (no leakage)

text

### Top-K choice

- **Top-10 làm primary** để có đủ pairs cho consistency metric (C(10,2) = 45 pairs)
- **Sensitivity analysis** với top-5 và top-20 trong Appendix

---

## 💡 Insights nổi bật (dùng cho Discussion)

### 1. Số anomalies thực sự trong test rất ít

Với mỗi fold, chỉ có **2–7 test samples** vượt `theta` (train threshold).
Nghĩa là top-10 đang explain phần lớn là normal samples.

→ Đây là **hạn chế của anomaly detection không nhãn**. Paper cần note
rõ điều này và justify chọn top-K bằng "consistency comparison" hơn là
"anomaly classification".

### 2. Method choice phụ thuộc vào device type

Kết quả walk-forward (trên dataset cũ) cho thấy:

| Device type | Method tốt nhất |
|-------------|:---:|
| Continuous (fridge) | SHAP-CA |
| Intermittent (printer) | LIME-CA |
| Kernel-based (OCSVM) | CHỈ SHAP-CA |

→ **Không có "one-size-fits-all"**. Cần decision rule cho practitioners.

### 3. OCSVM không explainable bằng LIME

Consistency của LIME-based methods trên OCSVM:
- Có thể âm hoặc gần 0
- Do RBF kernel tạo decision boundary phi tuyến
- Local linear surrogates không capture được

→ **SHAP-CA là lựa chọn duy nhất** cho kernel-based detectors.

---

## 📋 Checklist trước khi viết paper

- [x] Chọn dataset: CUBEMS (thay vì Sharjah)
- [x] Chọn zones: 6 zones (2 floors × 3 zones)
- [x] Feature engineering: power + 3-4 context features
- [x] Methodology: walk-forward 3 folds
- [ ] Viết `data_prep_cubems.py` (resample + extract features)
- [ ] Adapt `anomaly_detection.py`, `explainers.py`, `metrics.py` cho CUBEMS
- [ ] Chạy walk-forward đầy đủ
- [ ] Viết Section 3 (Methodology) + Section 4 (Setup)
- [ ] Viết lại Section 5 (Results) + Section 6 (Discussion)

---

## 🎯 Cần hỏi thầy

1. **Lux có nên là context feature không?** — Nếu có, cần defend tại sao
   (ánh sáng ảnh hưởng đến behavior của thiết bị).

2. **Alignment Score** (human vs device anomaly) có giữ không? — CUBEMS
   không có occupancy sensor, khó làm chặt.

3. **Số zones:** 6 zones đã đủ chưa, hay cần mở rộng ra thêm floors?

---

*Ghi chú được cập nhật lần cuối: 2026-09-19*
Lưu file vào Cubem/Note.md:

powershell
# Trong VS Code: Ctrl+N → paste → Ctrl+S → lưu tên "Note.md" trong folder Cubem/
Verify:

powershell
Get-Item "Note.md" | Select-Object Name, Length, LastWriteTime

python -c "import pandas as pd; df = pd.read_csv('results/frac_sensitivity_full_4metrics.csv'); print(df.groupby(['method', 'frac'])[['consistency', 'stability', 'fidelity', 'sensitivity']].agg(['mean', 'std']).round(4))"