"""
metrics.py
==========
Four evaluation metrics for local explanations (paper Section 4.4).

Definitions
-----------
1. Stability (CV)      : CoV of |phi_j| across R repetitions of the same
                         (detector, idx, method). Lower = better.
2. Fidelity (MSE)      : MSE between black-box f and linear surrogate g
                         on the 50 nearest neighbours of the explained
                         point. Lower = better.
3. Consistency (cosine): Mean pairwise cosine similarity of phi vectors
                         across all top-20 anomalies. Higher = better.
4. Sensitivity         : Absolute change in anomaly score after zeroing
                         the top-3 most important features. Higher = better.

All metrics take a `seed` and are deterministic given fixed input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


# ============================================================
# Metric container
# ============================================================

@dataclass
class MetricBundle:
    stability: float         # CV of |phi| across repetitions
    fidelity: float          # MSE on 50 nearest neighbours
    consistency: float       # mean pairwise cosine similarity
    sensitivity: float       # mean |Δscore| from zeroing top-3 features

    def as_dict(self) -> dict:
        return {
            "stability": self.stability,
            "fidelity": self.fidelity,
            "consistency": self.consistency,
            "sensitivity": self.sensitivity,
        }


# ============================================================
# 1. Stability (CV)
# ============================================================

def stability_cv(phi_runs: list[np.ndarray]) -> float:
    """
    Coefficient of variation of |phi_j| across R repetitions.

    Parameters
    ----------
    phi_runs : list of np.ndarray, each shape (d,)
        Explanations from the same (detector, idx, method) with different
        random seeds.

    Returns
    -------
    cv : float
        Mean over features j of (std_j / (mean_j + eps)).
    """
    if len(phi_runs) < 2:
        return 0.0
    P = np.stack([np.abs(p) for p in phi_runs], axis=0)  # (R, d)
    means = P.mean(axis=0)
    stds = P.std(axis=0, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv_j = np.where(means > 1e-12, stds / means, 0.0)
    return float(cv_j.mean())


# ============================================================
# 3. Consistency (cosine similarity)
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def consistency_mean_pairwise(phi_list: list[np.ndarray]) -> float:
    """
    Mean pairwise cosine similarity across all phi vectors (e.g. the
    explanation of all top-20 anomalies).
    """
    n = len(phi_list)
    if n < 2:
        return 0.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(phi_list[i], phi_list[j]))
    return float(np.mean(sims))


def fidelity_mse(
    predict_fn,
    X_eval_pool: np.ndarray,
    x_target: np.ndarray,
    phi: np.ndarray,
    n_neighbors: int = 50,
) -> float:
    """
    MSE between black-box f and linear surrogate g on n_neighbors.
    X_eval_pool: Chỉ chứa dữ liệu hợp lệ để tìm láng giềng (ví dụ: X_train hoặc quá khứ).
    x_target: Điểm dị biệt cần giải thích (1, d).
    """
    f_i_raw = float(predict_fn(x_target[None, :])[0])

    # Tìm 50 láng giềng gần nhất trong không gian đối chứng hợp lệ
    distances = np.linalg.norm(X_eval_pool - x_target, axis=1)
    
    # Nếu x_target nằm trong pool thì bỏ chính nó, nếu không thì lấy từ 0
    k = min(n_neighbors, len(X_eval_pool))
    order = np.argsort(distances)[:k]
    X_nn = X_eval_pool[order]

    # Raw predictions
    y_true_raw = predict_fn(X_nn)

    # Surrogate prediction
    y_pred_raw = (X_nn - x_target) @ phi + f_i_raw

    # Joint normalization
    all_vals = np.concatenate([y_true_raw, [f_i_raw]])
    s_min = float(all_vals.min())
    s_max = float(all_vals.max())
    scale = s_max - s_min
    if scale < 1e-12:
        return 0.0

    y_true_norm = (y_true_raw - s_min) / scale
    y_pred_norm = (y_pred_raw - s_min) / scale

    return float(np.mean((y_true_norm - y_pred_norm) ** 2))


def sensitivity_top3(
    predict_fn,
    X_eval_pool: np.ndarray,
    x_target: np.ndarray,
    phi: np.ndarray,
    top_k: int = 3,
    n_neighbors: int = 50,
) -> float:
    """
    Feature ablation sensitivity. Medians và Scale tính từ X_eval_pool.
    """
    f_orig = float(predict_fn(x_target[None, :])[0])

    # Local scale từ lân cận
    distances = np.linalg.norm(X_eval_pool - x_target, axis=1)
    k = min(n_neighbors, len(X_eval_pool))
    order = np.argsort(distances)[:k]
    y_neighbors = predict_fn(X_eval_pool[order])
    
    s_min = float(min(y_neighbors.min(), f_orig))
    s_max = float(max(y_neighbors.max(), f_orig))
    scale = s_max - s_min if (s_max - s_min) > 1e-12 else 1.0

    # Medians tính từ pool đã biết (Train), không tính trên toàn bộ tương lai
    medians = np.median(X_eval_pool, axis=0)

    top_features = np.argsort(np.abs(phi))[-top_k:]
    x_mod = x_target.copy()
    x_mod[top_features] = medians[top_features]

    f_mod = float(predict_fn(x_mod[None, :])[0])

    return float(abs(f_orig - f_mod) / scale)


def compute_metrics(
    predict_fn,
    X: np.ndarray,
    anomaly_indices: list[int],
    phi_runs_per_anomaly: dict[int, list[np.ndarray]],
    eval_pool_idx: np.ndarray | None = None,
    n_neighbors: int = 50,
) -> MetricBundle:
    """
    eval_pool_idx: Mảng chỉ số các dòng được phép dùng để tìm láng giềng.
    Nếu None, fallback về toàn bộ X (như cũ).
    """
    X_eval_pool = X[eval_pool_idx] if eval_pool_idx is not None else X

    stabs, fids, sems = [], [], []
    for idx in anomaly_indices:
        runs = phi_runs_per_anomaly.get(idx, [])
        if not runs:
            continue
        phi_first = runs[0]
        x_target = X[idx]

        stabs.append(stability_cv(runs))
        fids.append(fidelity_mse(predict_fn, X_eval_pool, x_target, phi_first,
                                 n_neighbors=n_neighbors))
        sems.append(sensitivity_top3(predict_fn, X_eval_pool, x_target, phi_first,
                                     top_k=3, n_neighbors=n_neighbors))

    phis = [runs[0] for idx in anomaly_indices
            if (runs := phi_runs_per_anomaly.get(idx, []))]
    cons = consistency_mean_pairwise(phis)

    return MetricBundle(
        stability=float(np.mean(stabs)) if stabs else 0.0,
        fidelity=float(np.mean(fids)) if fids else 0.0,
        consistency=float(cons),
        sensitivity=float(np.mean(sems)) if sems else 0.0,
    )

# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_prep import load_device_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS
    from anomaly_detection import (
        build_detectors, threshold_95, get_top_anomalies, get_normal_indices,
    )
    from explainers import explain_anomaly, get_predict_fn

    print("=" * 72)
    print("SMOKE TEST — metrics.py")
    print("=" * 72)

    device = "fridge"
    df = load_device_dataframe(device, verbose=False)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

    dets = build_detectors(X, ["IF"])
    det = dets["IF"]
    predict_fn = get_predict_fn(det, "IF")

    scores = det.score(X)
    theta = threshold_95(scores)
    normal_idx = get_normal_indices(scores, theta)
    top20 = get_top_anomalies(scores, k=20).tolist()

    # Test all three methods
    for method in ["SHAP-CA", "LIME-std", "LIME-CA"]:
        print(f"\n{'-'*72}")
        print(f"Method: {method}")
        print(f"{'-'*72}")

        # Run 3 repetitions per anomaly for stability
        phi_runs = {}
        for idx in top20:
            runs = []
            for rep in range(10):
                res = explain_anomaly(
                    det, "IF", X, idx, normal_idx,
                    method, device=device,
                    context_idx=context_idx,
                    seed=42 + rep,
                )
                runs.append(res.phi)
            phi_runs[idx] = runs

        bundle = compute_metrics(predict_fn, X, top20, phi_runs)
        print(f"  Stability (CV, lower=better)     : {bundle.stability:.4f}")
        print(f"  Fidelity  (MSE, lower=better)    : {bundle.fidelity:.6f}")
        print(f"  Consistency (cos, higher=better) : {bundle.consistency:.4f}")
        print(f"  Sensitivity (Δscore, higher=better): {bundle.sensitivity:.4f}")

    print("\n[DONE] Smoke test complete.")