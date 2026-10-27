"""
ablation_context_selection.py
=============================
Ablation Study phân rã đóng góp của các thành phần trong Context Selection (LIME-CA):

  1. Fixed-Cosine (Baseline sơ khai):
     - Đo khoảng cách cosine thô trên toàn bộ vector context chuẩn hóa.
     - Không có trọng số phân khối, không mã hóa chu kỳ tuần hoàn.

  2. Geom-Fixed (Bậc trung):
     - Mã hóa chu kỳ tuần hoàn (chord distance cho giờ, circular distance cho ngày).
     - Kết hợp khoảng cách Euclidean với trọng số cố định đồng đều giữa các khối.

  3. Geom-Learned (Đầy đủ - Đề xuất):
     - Kế thừa toàn bộ cấu trúc hình học tuần hoàn của Geom-Fixed.
     - Sử dụng trọng số học máy thích ứng theo mô hình (Learned Weights) tính 
       từ độ nhạy hoán vị trên tập Train-Normal và khóa cố định trước khi đánh giá Test.

Đảm bảo phương pháp luận:
  - Khớp độ dài động theo từng Zone (không tràn index).
  - Khóa chặt Train-Scaler và Validation-Threshold.
  - Đánh giá Out-of-sample sạch (eval_pool_idx chỉ lấy từ Train).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep_cubems import (
    load_zone_dataframe,
    FEATURE_COLUMNS,
    CONTEXT_COLUMNS,
    ZONE_IDS,
)
from anomaly_detection import build_detectors
import explainers
from explainers import (
    explain_anomaly,
    get_predict_fn,
    compute_global_importance,
    compute_context_weights,
)
from metrics import compute_metrics
from walk_forward_cubems import build_folds_for_length


# ============================================================
# CẤU HÌNH THỰC NGHIỆM CHUẨN
# ============================================================

FRAC = 0.40
MIN_K = 50

# Trọng số trực quan cố định chia đều theo 5 khối ngữ cảnh
FIXED_WEIGHTS = {
    "hour": 0.25,
    "day": 0.15,
    "temperature": 0.20,
    "humidity": 0.20,
    "lux": 0.20,
}


# ============================================================
# HELPER CHỌN TOP-K
# ============================================================

def _select_top_k(dists, normal_indices, frac=FRAC, min_k=MIN_K):
    """Giữ nguyên kích thước K láng giềng giữa tất cả cấu hình để cô lập biến số."""
    dists = np.asarray(dists, dtype=float)
    normal_indices = np.asarray(normal_indices, dtype=int)
    n = len(normal_indices)

    if n == 0:
        raise ValueError("normal_indices is empty.")

    k = min(max(min_k, int(np.ceil(frac * n))), n)
    order = np.argsort(dists, kind="stable")[:k]
    return normal_indices[order]


# ============================================================
# 1. CẤU HÌNH 1: NAIVE FIXED COSINE (BASELINE)
# ============================================================

def pool_cosine_naive(
    c_i,
    C_normal,
    normal_indices,
    frac=FRAC,
    min_k=MIN_K,
    **kwargs,
):
    """
    Cosine distance thô không trọng số trên toàn bộ vector context.
    Thể hiện cách tiếp cận cơ bản ban đầu khi chưa có xử lý chu kỳ hay học trọng số.
    """
    c_i = np.asarray(c_i, dtype=float)
    C_normal = np.asarray(C_normal, dtype=float)

    norm_c = np.linalg.norm(c_i) + 1e-12
    norm_C = np.linalg.norm(C_normal, axis=1) + 1e-12

    cos_sim = (C_normal @ c_i) / (norm_C * norm_c)
    dists = 1.0 - cos_sim

    return _select_top_k(dists, normal_indices, frac=frac, min_k=min_k)


# ============================================================
# HÀM LÕI HÌNH HỌC TUẦN HOÀN (CHORD + CIRCULAR EUCLIDEAN)
# ============================================================

def _pool_geom_core(
    c_i,
    C_normal,
    normal_indices,
    weights,
    frac=FRAC,
    min_k=MIN_K,
):
    c_i = np.asarray(c_i, dtype=float)
    C_normal = np.asarray(C_normal, dtype=float)

    # 1. Giờ: Khôi phục về vòng tròn đơn vị [-1, 1] và tính chord distance
    sin_normal = 2.0 * C_normal[:, 0] - 1.0
    cos_normal = 2.0 * C_normal[:, 1] - 1.0
    sin_i = 2.0 * c_i[0] - 1.0
    cos_i = 2.0 * c_i[1] - 1.0

    d_hour = 0.5 * np.sqrt((sin_normal - sin_i) ** 2 + (cos_normal - cos_i) ** 2)

    # 2. Ngày: Khôi phục về [0, 6] và tính khoảng cách vòng tuần hoàn 7 ngày
    day_normal = C_normal[:, 2] * 6.0
    day_i = c_i[2] * 6.0
    day_diff = np.abs(day_normal - day_i)
    day_diff = np.minimum(day_diff, 7.0 - day_diff)
    d_day = day_diff / 3.5  # Chuẩn hóa về [0, 1]

    # 3. Môi trường: Các biến đã nằm sẵn trong dải [0, 1]
    d_temp = np.abs(C_normal[:, 3] - c_i[3])
    d_humid = np.abs(C_normal[:, 4] - c_i[4])
    d_lux = np.abs(C_normal[:, 5] - c_i[5])

    # 4. Trọng số kết hợp
    dists_sq = (
        weights["hour"] * (d_hour ** 2)
        + weights["day"] * (d_day ** 2)
        + weights["temperature"] * (d_temp ** 2)
        + weights["humidity"] * (d_humid ** 2)
        + weights["lux"] * (d_lux ** 2)
    )

    dists = np.sqrt(dists_sq)
    return _select_top_k(dists, normal_indices, frac=frac, min_k=min_k)


# ============================================================
# 2. CẤU HÌNH 2: GEOM-FIXED (BẬC TRUNG)
# ============================================================

def pool_geom_fixed(
    c_i,
    C_normal,
    normal_indices,
    frac=FRAC,
    min_k=MIN_K,
    context_weights=None,
):
    """Mã hóa chu kỳ tuần hoàn kết hợp khoảng cách Euclidean với trọng số cố định."""
    return _pool_geom_core(
        c_i, C_normal, normal_indices, FIXED_WEIGHTS, frac=frac, min_k=min_k
    )


# ============================================================
# 3. CẤU HÌNH 3: GEOM-LEARNED (ĐẦY ĐỦ - ĐỀ XUẤT)
# ============================================================

def pool_geom_learned(
    c_i,
    C_normal,
    normal_indices,
    frac=FRAC,
    min_k=MIN_K,
    context_weights=None,
):
    """Mã hóa chu kỳ tuần hoàn kết hợp trọng số học máy thích ứng theo mô hình."""
    weights = context_weights if context_weights is not None else FIXED_WEIGHTS
    return _pool_geom_core(
        c_i, C_normal, normal_indices, weights, frac=frac, min_k=min_k
    )


# ============================================================
# TIẾN TRÌNH CHẠY THỰC NGHIỆM ABLATION
# ============================================================

def run_ablation(
    zones: list[str],
    detectors: list[str] = ["IF"],
    reps: int = 3,
    top_k: int = 20,
    seed: int = 42,
):
    configs = [
        "Fixed-Cosine",
        "Geom-Fixed",
        "Geom-Learned",
    ]

    orig_select_context_pool = explainers.select_context_pool
    all_rows = []

    try:
        for zone_id in zones:
            print(f"\n{'='*72}\nĐang chạy Ablation cho Zone: {zone_id}...\n{'='*72}")

            df = load_zone_dataframe(zone_id, verbose=False)
            X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
            timestamps = df.index

            # Tự động tạo mốc folds theo số dòng thực tế của file
            folds = build_folds_for_length(len(X_raw))

            for fold in folds:
                print(f"\n--- Fold {fold.idx} ({fold.describe(timestamps)}) ---")

                # 1. Fit Scaler CHỈ trên Train (chống rò rỉ dữ liệu)
                scaler = MinMaxScaler(feature_range=(0.0, 1.0))
                scaler.fit(X_raw[:fold.train_end])
                X_scaled = np.clip(scaler.transform(X_raw), 0.0, 1.0)

                X_train = X_scaled[:fold.train_end]
                X_val = X_scaled[fold.train_end:fold.val_end]
                X_test = X_scaled[fold.val_end:fold.test_end]

                # 2. Huấn luyện Detector
                det_models = build_detectors(
                    X_train, detectors, X_val=X_val, random_state=seed
                )

                for det_name, det in det_models.items():
                    scores_all = det.score(X_scaled)
                    scores_train = scores_all[:fold.train_end]
                    scores_val = scores_all[fold.train_end:fold.val_end]
                    scores_test = scores_all[fold.val_end:fold.test_end]

                    # 3. Lấy ngưỡng Validation và Train Normal Pool
                    theta = float(np.percentile(scores_val, 95))
                    theta_train = float(np.percentile(scores_train, 95))

                    train_indices = np.arange(0, fold.train_end)
                    normal_idx_global = train_indices[scores_train <= theta_train]

                    # 4. Lấy mẫu Test Anomalies phân tầng (Stratified Quantile)
                    flagged_local = np.where(scores_test > theta)[0]
                    test_global_indices = np.arange(fold.val_end, fold.test_end)

                    if len(flagged_local) == 0:
                        top_k_local = np.argsort(scores_test)[-top_k:][::-1]
                    else:
                        sorted_flagged = flagged_local[np.argsort(scores_test[flagged_local])[::-1]]
                        stride = np.linspace(
                            0, len(sorted_flagged) - 1, min(top_k, len(sorted_flagged)), dtype=int
                        )
                        top_k_local = sorted_flagged[stride]

                    top_k_global = test_global_indices[top_k_local]
                    predict_fn = get_predict_fn(det, det_name)

                    # 5. Học trọng số ngữ cảnh CHỈ từ Train-Normal (Khóa trước khi sang Test)
                    X_train_normal = X_scaled[normal_idx_global]
                    global_imp = compute_global_importance(
                        predict_fn, X_train_normal, n_repeats=5, seed=seed
                    )
                    learned_weights = compute_context_weights(
                        global_imp, FEATURE_COLUMNS, CONTEXT_COLUMNS
                    )

                    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

                    # 6. Chạy đánh giá từng Cấu hình Ablation
                    for cfg in configs:
                        t0 = time.perf_counter()

                        if cfg == "Fixed-Cosine":
                            explainers.select_context_pool = pool_cosine_naive
                            active_weights = FIXED_WEIGHTS.copy()
                        elif cfg == "Geom-Fixed":
                            explainers.select_context_pool = pool_geom_fixed
                            active_weights = FIXED_WEIGHTS.copy()
                        else:  # Geom-Learned
                            explainers.select_context_pool = pool_geom_learned
                            active_weights = learned_weights

                        phi_runs = {}
                        for g_idx in top_k_global:
                            g_idx = int(g_idx)
                            runs = []
                            for rep in range(reps):
                                res = explain_anomaly(
                                    det,
                                    det_name,
                                    X=X_scaled,
                                    idx=g_idx,
                                    normal_idx=normal_idx_global,
                                    method="LIME-CA",
                                    device=zone_id,
                                    context_idx=context_idx,
                                    context_weights=active_weights,
                                    seed=seed + rep,
                                )
                                runs.append(res.phi)
                            phi_runs[g_idx] = runs

                        # TRIỆT TIÊU EVALUATION LEAKAGE: Chỉ dùng tập Train làm pool đánh giá
                        eval_pool_indices = np.arange(0, fold.train_end)

                        bundle = compute_metrics(
                            predict_fn,
                            X_scaled,
                            top_k_global.tolist(),
                            phi_runs,
                            eval_pool_idx=eval_pool_indices,
                        )

                        elapsed = time.perf_counter() - t0
                        print(
                            f"  [{cfg:14s}] "
                            f"cons={bundle.consistency:.4f}  "
                            f"stab={bundle.stability:.4f}  "
                            f"fid={bundle.fidelity:.6f}  "
                            f"sens={bundle.sensitivity:.4f}  "
                            f"({elapsed:.1f}s)"
                        )

                        all_rows.append({
                            "zone": zone_id,
                            "fold": fold.idx,
                            "detector": det_name,
                            "config": cfg,
                            "consistency": bundle.consistency,
                            "stability": bundle.stability,
                            "fidelity": bundle.fidelity,
                            "sensitivity": bundle.sensitivity,
                        })

    finally:
        # Luôn khôi phục hàm gốc sau khi chạy xong
        explainers.select_context_pool = orig_select_context_pool

    # ============================================================
    # TỔNG HỢP VÀ IN BÁO CÁO BẬC THANG CẢI TIẾN
    # ============================================================
    df_res = pd.DataFrame(all_rows)

    print("\n" + "=" * 72)
    print("ABLATION STUDY: BẢNG TỔNG HỢP TOÀN CỤC (MEAN ± STD)")
    print("=" * 72)
    agg = df_res.groupby("config")[
        ["consistency", "stability", "fidelity", "sensitivity"]
    ].agg(["mean", "std"]).round(4)
    print(agg)

    print("\n" + "=" * 72)
    print("BẬC THANG CẢI TIẾN TUẦN TỰ (SEQUENTIAL DELTAS)")
    print("=" * 72)
    mean_table = df_res.groupby("config")[
        ["consistency", "stability", "fidelity", "sensitivity"]
    ].mean()

    steps = [
        ("Fixed-Cosine", "Geom-Fixed"),
        ("Geom-Fixed", "Geom-Learned"),
    ]

    for a, b in steps:
        delta = mean_table.loc[b] - mean_table.loc[a]
        print(f"\n{a} -> {b}:")
        print(f"  Δ Consistency = {delta['consistency']:+.4f} (Higher is better)")
        print(f"  Δ Stability   = {delta['stability']:+.4f} (Lower is better)")
        print(f"  Δ Fidelity    = {delta['fidelity']:+.6f} (Lower is better)")
        print(f"  Δ Sensitivity = {delta['sensitivity']:+.4f} (Higher is better)")

    out_path = Path("results/ablation_context_selection.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"\n[Saved] Kết quả chi tiết đã được lưu tại: {out_path}")

    return df_res


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Chạy trên toàn bộ 6 zones đúng theo yêu cầu của bài báo
    run_ablation(
        zones=ZONE_IDS,
        detectors=["IF"],
        reps=10,
        top_k=20,
        seed=42,
    )