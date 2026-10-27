# CU-BEMS Anomaly Detection & Explainable AI (XAI) Framework

Dự án nghiên cứu và triển khai hệ thống phát hiện bất thường (Anomaly Detection) kết hợp giải thích mô hình AI (XAI với LIME-CA và SHAP-CA) trên bộ dữ liệu quản lý năng lượng tòa nhà **CU-BEMS (Chulalongkorn University Building Energy Management System)**.

---

## 📌 Tổng quan dự án (Overview)

Mục tiêu chính của dự án là xây dựng và đánh giá hiệu năng của các mô hình phát hiện bất thường trong dữ liệu chuỗi thời gian (time-series) của hệ thống HVAC, đồng thời sử dụng các phương pháp XAI nâng cao để giải thích nguyên nhân gây ra bất thường ở cấp độ vùng (zone-level) và tầng (floor-level).

### Key Features:
* **Data Pipeline:** Xử lý và chuẩn hóa dữ liệu năng lượng từ bộ dữ liệu thô `CU-BEMS`.
* **Anomaly Detection:** Phát hiện bất thường bằng chiến lược đánh giá kiểm thử cuốn chiếu (Walk-Forward Validation).
* **Explainable AI (XAI):** Tích hợp và tinh chỉnh các phương pháp giải thích mô hình (LIME-CA, SHAP-CA).
* **Domain Validation & Sensitivity:** Đánh giá tính nhất quán, độ tin cậy của lời giải thích (Attribution Plausibility) và phân tích độ nhạy (Sensitivity Analysis).

---

## 📁 Cấu trúc thư mục (Repository Structure)

```text
├── data/
│   ├── raw/                  # Dữ liệu thô CU-BEMS (2019Floor2.csv, 2019Floor5.csv,...)
│   └── verify_overlap_impact.py
├── src/
│   ├── check/                # Các script kiểm tra chỉ số và fold index
│   │   ├── export_fold_indices.py
│   │   └── verify_table3.py
│   ├── make_figure/          # Các script vẽ biểu đồ trực quan hóa
│   │   ├── generate_figures.py
│   │   ├── plot_ablation.py
│   │   ├── plot_cross_zone_consistency.py
│   │   ├── plot_figure_domain_validation.py
│   │   ├── plot_frac_sensitivity.py
│   │   └── plot_threshold_sensitivity.py
│   ├── ablation_context_selection.py
│   ├── anomaly_detection.py  # Thuật toán phát hiện bất thường
│   ├── data_prep_cubems.py   # Tiền xử lý dữ liệu CU-BEMS
│   ├── explainers.py         # Cấu hình các bộ giải thích XAI (LIME-CA, SHAP-CA)
│   ├── metrics.py            # Định nghĩa các chỉ số đánh giá (F1, Precision, Recall,...)
│   ├── run_case_study.py
│   ├── run_domain_validation_groundtruth.py
│   ├── threshold_sensitivity.py
│   ├── tune_fidelity_neighborhood.py
│   ├── tune_frac_sensitivity.py
│   └── walk_forward_cubems.py # Đánh giá theo mô hình Walk-Forward
└── note.md                   # Ghi chú quá trình thực nghiệm
