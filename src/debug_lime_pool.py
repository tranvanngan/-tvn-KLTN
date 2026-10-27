"""
debug_lime_pool.py
==================
Test 3 variants của LIME-CA pool selection TRƯỚC khi sửa chính thức.

Variants
--------
A (current):  top p% context-similar normal points (p=0.40)
B (tau-only): normal points với sim(c_i, c_j) >= tau (tau=0.5)
C (hybrid):   tau filter + cap pool size về [MIN_POOL, MAX_POOL]

Chạy trên:
  - Zone: floor2_z1
  - Fold 1: Train=[0:4380], Test=[5110:6570]
  - Detector: IF
  - Top-K test anomalies: 20
  - Reps: 5 (để tính stability)

Output:
  - Bảng so sánh 3 variants: consistency, stability, fidelity
  - Pool size thực tế của mỗi variant
"""
from __future__ import annotations
from sklearn.preprocessing import MinMaxScaler
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep_cubems import (
    load_zone_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS,
)
from anomaly_detection import build_detectors
from explainers import (
    get_predict_fn,
    _cosine_sim,
    select_context_pool,
)
from metrics import (
    stability_cv, fidelity_mse, consistency_mean_pairwise,
    sensitivity_top3,
)


# ============================================================
# Config
# ============================================================

ZONE_ID = "floor2_z1"
DETECTOR = "IF"
FOLD_TRAIN_END = 4380
FOLD_VAL_END = 5110
FOLD_TEST_END = 6570

TOP_K = 20
REPS = 5
SEED = 42

M = 150
SIGMA = 0.1
KERNEL_WIDTH = 0.75
RIDGE_ALPHA = 1.0

# Variant A params
P_FRACTION = 0.40

# Variant B/C params
TAU = 0.5
MIN_POOL = 50
MAX_POOL = 200


# ============================================================
# 3 Variants của LIME-CA
# ============================================================

def _lime_ca_core(
    detector, detector_name, X, idx, normal_idx, context_idx,
    pool_indices, M, sigma, kernel_width, ridge_alpha, seed,
):
    """
    Core LIME-CA: sample M perturbations từ pool_indices,
    thêm noise vào non-context features, fit Ridge.
    """
    rng = np.random.default_rng(seed)
    predict_fn = get_predict_fn(detector, detector_name)

    x_i = X[idx].copy()
    d = X.shape[1]

    # Sample base indices with replacement từ pool
    base_indices = rng.choice(pool_indices, size=M, replace=True)
    Z = X[base_indices].copy()

    # Noise chỉ vào non-context features
    context_set = set(context_idx)
    device_idx = [j for j in range(d) if j not in context_set]
    if device_idx:
        noise = rng.normal(0.0, sigma, size=(M, len(device_idx)))
        Z[:, device_idx] += noise
    Z = np.clip(Z, 0.0, 1.0)


# Weighted Ridge surrogate
    y = predict_fn(np.vstack([Z, x_i[None, :]]))[:M]
    distances = np.linalg.norm(Z - x_i[None, :], axis=1)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    
    # Chống trường hợp underflow về 0 toàn bộ mảng
    if weights.max() < 1e-12:
        weights = np.ones_like(weights)
    else:
        weights = weights / (weights.max() + 1e-12)

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(Z, y, sample_weight=weights)
    return ridge.coef_.copy(), len(pool_indices)

def lime_ca_variant_A(
    detector, detector_name, X, idx, normal_idx, context_idx,
    M=150, sigma=0.1, kernel_width=0.75, ridge_alpha=1.0,
    p=0.40, seed=42,
):
    """Variant A: cosine similarity, top p%."""

    x_i = X[idx]
    c_i = x_i[context_idx]
    C = X[normal_idx][:, context_idx]

    sims = _cosine_sim(c_i, C)

    n_top = max(1, int(np.ceil(p * len(normal_idx))))
    top_local = np.argsort(sims)[-n_top:]
    pool = normal_idx[top_local]

    phi, pool_size = _lime_ca_core(
        detector, detector_name, X, idx, normal_idx, context_idx,
        pool, M, sigma, kernel_width, ridge_alpha, seed,
    )

    return phi, pool_size

def lime_ca_variant_B(
    detector, detector_name, X, idx, normal_idx, context_idx,
    M=150, sigma=0.1, kernel_width=0.75, ridge_alpha=1.0,
    p=0.40, seed=42,
):
    """Variant B: cyclic z-score context pool, top p%."""

    x_i = X[idx]
    c_i = x_i[context_idx]
    C = X[normal_idx][:, context_idx]

    pool = select_context_pool(
        c_i=c_i,
        C_normal=C,
        normal_indices=normal_idx,
        frac=p,
        min_k=50,
    )

    phi, pool_size = _lime_ca_core(
        detector, detector_name, X, idx, normal_idx, context_idx,
        pool, M, sigma, kernel_width, ridge_alpha, seed,
    )

    return phi, pool_size

def diagnose_feature_distance(
    X,
    idx,
    normal_idx,
    pool_A,
    pool_B,
    context_idx,
):
    
    """
    Compare feature-wise squared z-distance contributions
    using a COMMON reference mean/std computed from the full
    normal training pool.
    """

    x_i = X[idx]
    C_full = X[normal_idx][:, context_idx]
    C_A = X[pool_A][:, context_idx]
    C_B = X[pool_B][:, context_idx]
    c_i = x_i[context_idx]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use the SAME reference distribution for A and B.
    # ---------------------------------------------------------
    mu = C_full.mean(axis=0)
    sd = np.maximum(C_full.std(axis=0), 1e-6)

    c_z = (c_i - mu) / sd
    A_z = (C_A - mu) / sd
    B_z = (C_B - mu) / sd

    # Squared z-distance contribution per feature
    A_contrib = np.mean((A_z - c_z[None, :]) ** 2, axis=0)
    B_contrib = np.mean((B_z - c_z[None, :]) ** 2, axis=0)

    feature_names = [
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "temperature",
        "humidity",
        "lux",
    ]

    print("\nFEATURE-WISE POOL DISTANCE DIAGNOSTIC")
    print("Common z-score reference: full normal training pool")
    print("Mean squared z-distance contribution\n")

    print(
        f"{'Feature':<18}"
        f"{'A: Cosine':>15}"
        f"{'B: Context':>15}"
        f"{'B-A':>15}"
    )
    print("-" * 63)

    for name, a, b in zip(feature_names, A_contrib, B_contrib):
        print(
            f"{name:<18}"
            f"{a:>15.6f}"
            f"{b:>15.6f}"
            f"{(b - a):>15.6f}"
        )
# ============================================================
# Metrics per variant
# ============================================================

def evaluate_variant(
    name, variant_fn, detector, detector_name, X, test_anomalies,
    normal_idx, context_idx, reps, seed,
):
    """Chạy variant trên tất cả anomalies, tính 4 metrics."""
    print(f"\n{'='*72}")
    print(f"VARIANT: {name}")
    print(f"{'='*72}")

    phi_runs_per_anomaly = {}
    pool_sizes = []
    runtimes = []

    for idx in test_anomalies:
        idx = int(idx)
        runs = []
        for rep in range(reps):
            t0 = time.perf_counter()
            phi, pool_size = variant_fn(
                detector, detector_name, X, idx, normal_idx, context_idx,
                seed_=seed + rep,
            )
            runtimes.append((time.perf_counter() - t0) * 1000)
            runs.append(phi)
        phi_runs_per_anomaly[idx] = runs
        pool_sizes.append(pool_size)

    # Stability
    stab = np.mean([
        stability_cv(phi_runs_per_anomaly[idx]) for idx in test_anomalies
    ])

    # Fidelity (dùng phi của rep đầu tiên)
    predict_fn = get_predict_fn(detector, detector_name)
    fids = []
    for idx in test_anomalies:
        idx = int(idx)
        fid = fidelity_mse(predict_fn, X, idx, phi_runs_per_anomaly[idx][0])
        fids.append(fid)
    fid_mean = np.mean(fids)

    # Consistency
    phis_first = [phi_runs_per_anomaly[int(idx)][0] for idx in test_anomalies]
    cons = consistency_mean_pairwise(phis_first)

    # Sensitivity
    sens = np.mean([
        sensitivity_top3(predict_fn, X, int(idx),
                         phi_runs_per_anomaly[int(idx)][0])
        for idx in test_anomalies
    ])

    print(f"  Pool size:       min={min(pool_sizes):4d}  "
          f"max={max(pool_sizes):4d}  "
          f"mean={np.mean(pool_sizes):7.1f}")
    print(f"  Runtime:         {np.mean(runtimes):.1f} ms/explanation")
    print(f"  Stability (CV):  {stab:.4f}  (lower better)")
    print(f"  Fidelity (MSE):  {fid_mean:.6f}  (lower better)")
    print(f"  Consistency:     {cons:.4f}  (higher better)")
    print(f"  Sensitivity:     {sens:.4f}  (higher better)")

    return {
        "variant": name,
        "stability": stab,
        "fidelity": fid_mean,
        "consistency": cons,
        "sensitivity": sens,
        "pool_mean": float(np.mean(pool_sizes)),
        "runtime_ms": float(np.mean(runtimes)),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("DEBUG: SO SÁNH 2 CÁCH CHỌN LIME-CA POOL")    
    print("=" * 72)
    print(f"Zone:      {ZONE_ID}")
    print(f"Detector:  {DETECTOR}")
    print(f"Fold 1:    Train=[0:{FOLD_TRAIN_END}]  Test=[{FOLD_VAL_END}:{FOLD_TEST_END}]")
    print(f"Top-K:     {TOP_K}")
    print(f"Reps:      {REPS}")
    print(f"\nVariant A: cosine top-{P_FRACTION:.0%}")
    print(f"Variant B: select_context_pool top-{P_FRACTION:.0%}")

# Load data
    df = load_zone_dataframe(ZONE_ID, verbose=False)
    X_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]
    print(f"\nX shape: {X_raw.shape}")
    print(f"Context idx: {context_idx} -> "
          f"{[FEATURE_COLUMNS[i] for i in context_idx]}")

    # Scale X dựa trên train fold
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_raw[:FOLD_TRAIN_END])
    X = np.clip(scaler.transform(X_raw), 0.0, 1.0)

    # Fit detector trên train
    X_train = X[:FOLD_TRAIN_END]
    detectors = build_detectors(X_train, [DETECTOR], random_state=SEED)
    det = detectors[DETECTOR]
    # Score full X
    scores_all = det.score(X)
    theta = float(np.percentile(scores_all[:FOLD_TRAIN_END], 95))
    normal_idx = np.arange(0, FOLD_TRAIN_END)[
        scores_all[:FOLD_TRAIN_END] <= theta
    ]

    # Top-K test anomalies
    test_global = np.arange(FOLD_VAL_END, FOLD_TEST_END)
    scores_test = scores_all[FOLD_VAL_END:FOLD_TEST_END]
    top_k_local = np.argsort(scores_test)[-TOP_K:][::-1]
    test_anomalies = test_global[top_k_local]
    # =========================================================
    # DEBUG: Compare pool A vs B on ONE anomaly
    # =========================================================

    idx = int(test_anomalies[0])

    x_i = X[idx]
    c_i = x_i[context_idx]
    C = X[normal_idx][:, context_idx]

    # ----- Pool A: cosine top-40% -----
    sims_A = _cosine_sim(c_i, C)

    n_top = max(
        1,
        int(np.ceil(P_FRACTION * len(normal_idx)))
    )

    top_local_A = np.argsort(sims_A)[-n_top:]
    pool_A = normal_idx[top_local_A]

    # ----- Pool B: context-distance top-40% -----
    pool_B = select_context_pool(
        c_i=c_i,
        C_normal=C,
        normal_indices=normal_idx,
        frac=P_FRACTION,
        min_k=50,
    )
        # =========================================================
    # DEBUG: Distance of A-only vs B-only points
    # =========================================================
    A_only = np.setdiff1d(pool_A, pool_B)
    B_only = np.setdiff1d(pool_B, pool_A)

    print("\nA-only / B-only:")
    print(f"  A-only: {len(A_only)}")
    print(f"  B-only: {len(B_only)}")

    # Common z-score reference: full normal training pool
    C_full = X[normal_idx][:, context_idx]

    mu = C_full.mean(axis=0)
    sd = np.maximum(C_full.std(axis=0), 1e-6)

    c_z = (c_i - mu) / sd
    C_z = (C_full - mu) / sd

    # Map global index -> row in C_full
    normal_pos = {
        int(global_idx): pos
        for pos, global_idx in enumerate(normal_idx)
    }

    def get_z_distances(indices):
        if len(indices) == 0:
            return np.array([])

        rows = np.array([
            normal_pos[int(global_idx)]
            for global_idx in indices
        ])

        diff = C_z[rows] - c_z[None, :]

        # Cyclic distance for day_of_week
        if len(context_idx) >= 3:
            day_col = 2

            day_diff = np.abs(
                C_full[rows, day_col] - c_i[day_col]
            )
            day_diff = np.minimum(
                day_diff,
                7.0 - day_diff
            )

            day_sd = max(C_full[:, day_col].std(), 1e-6)
            day_distance = day_diff / day_sd

            diff[:, day_col] = day_distance

        return np.linalg.norm(diff, axis=1)
    

    d_A_only = get_z_distances(A_only)
    d_B_only = get_z_distances(B_only)

    print("\nContext z-distance to anomaly:")

    if len(d_A_only) > 0:
        print(
            f"  A-only: mean={d_A_only.mean():.4f}, "
            f"median={np.median(d_A_only):.4f}, "
            f"min={d_A_only.min():.4f}, "
            f"max={d_A_only.max():.4f}"
        )

    if len(d_B_only) > 0:
        print(
            f"  B-only: mean={d_B_only.mean():.4f}, "
            f"median={np.median(d_B_only):.4f}, "
            f"min={d_B_only.min():.4f}, "
            f"max={d_B_only.max():.4f}"
        )

    print("\n" + "=" * 72)
    print("POOL DEBUG — ONE ANOMALY")
    print("=" * 72)

    print(f"Anomaly index: {idx}")
    print(f"Pool A size:   {len(pool_A)}")
    print(f"Pool B size:   {len(pool_B)}")

    # Overlap giữa hai pool
    overlap = len(set(pool_A) & set(pool_B))
    union = len(set(pool_A) | set(pool_B))

    print(f"Overlap:       {overlap}")
    print(f"Jaccard:       {overlap / union:.4f}")

    # Context của anomaly
    print("\nAnomaly context:")
    for name, value in zip(
        [FEATURE_COLUMNS[i] for i in context_idx],
        c_i,
    ):
        print(f"  {name:15s}: {value:.4f}")

    # Context mean của hai pool
    print("\nMean context of selected pools:")

    mean_A = X[pool_A][:, context_idx].mean(axis=0)
    mean_B = X[pool_B][:, context_idx].mean(axis=0)

    for name, a, b in zip(
        [FEATURE_COLUMNS[i] for i in context_idx],
        mean_A,
        mean_B,
    ):
        print(
            f"  {name:15s}: "
            f"A={a:.4f}   B={b:.4f}"
        )

    print("=" * 72)
    print(f"\nNormal pool (train only): {len(normal_idx)} points")
    print(f"Test anomalies (top-{TOP_K}): {test_anomalies[:5].tolist()} ...")

    # Evaluate 3 variants
    results = []
  
    for name, fn in [
    ("A: cosine top-40%", lime_ca_variant_A),
    ("B: select_context_pool top-40%", lime_ca_variant_B),
    ]:
        def wrapped(det_, det_name_, X_, idx_, normal_, ctx_, seed_):
            if name.startswith("A"):
                return fn(
                    det_, det_name_, X_, idx_, normal_, ctx_,
                    p=P_FRACTION,
                    seed=seed_,
                )
            elif name.startswith("B"):
                return fn(
                    det_, det_name_, X_, idx_, normal_, ctx_,
                    p=P_FRACTION,
                    seed=seed_,
                )
            elif name.startswith("C"):
                return fn(
                    det_, det_name_, X_, idx_, normal_, ctx_,
                    tau=TAU,
                    min_pool=MIN_POOL,
                    max_pool=MAX_POOL,
                    seed=seed_,
                )
        r = evaluate_variant(
            name, wrapped, det, DETECTOR, X, test_anomalies,
            normal_idx, context_idx, REPS, SEED,
        )
        results.append(r)

    # So sánh
    print(f"\n{'='*72}")
    print("SO SÁNH 2 VARIANTS")    
    print(f"{'='*72}")
    print(f"{'Variant':28s}  {'Cons':>8s}  {'Stab':>8s}  "
          f"{'Fid':>10s}  {'Sens':>8s}  {'Pool':>7s}")
    print(f"{'-'*28}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}")
    for r in results:
        print(f"{r['variant']:28s}  {r['consistency']:8.4f}  "
              f"{r['stability']:8.4f}  {r['fidelity']:10.6f}  "
              f"{r['sensitivity']:8.4f}  {r['pool_mean']:7.1f}")

    print("=" * 72)
    print("DIAGNOSTIC RESULT")
    print("=" * 72)


    print("Both pool-selection variants completed successfully.")
