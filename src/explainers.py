"""
explainers.py
=============
Five local explanation methods for anomaly detectors in the CLEF pipeline.

Methods
-------
1. SHAP-CA   : KernelSHAP with context-aware background selection
2. LIME-std  : Standard LIME (isotropic Gaussian perturbations, fixed
              sigma, centered no-intercept Ridge surrogate)
3. LIME-CA   : Context-aware LIME (same sampling/regression scheme as
              LIME-std, but per-feature noise bandwidth is scaled by
              the ratio of context-local to global feature variability,
              so it degenerates to LIME-std when context adds no
              information -- NOT context-reuse/DLIME-style sampling)
4. DeepSHAP  : GradientExplainer with context-aware background (MLP only)
5. IG        : Integrated Gradients with median baseline (MLP only)

Unified interface
-----------------
    explain_anomaly(detector, X, idx, normal_idx, method, **kwargs)
        -> ExplanationResult(phi, runtime_ms, n_calls)

Reproducibility
---------------
All methods accept `seed`. Given same (detector, X, idx, normal_idx, method,
seed, config) -> identical phi (up to 1e-10 float tolerance).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Result container
# ============================================================

@dataclass
class ExplanationResult:
    method: str
    device: str
    detector: str
    anomaly_idx: int
    phi: np.ndarray                     # (d,) attribution vector
    runtime_ms: float
    n_calls: int                        # number of black-box calls
    meta: dict = field(default_factory=dict)

    def save(self, path):
        np.savez(
            path,
            phi=self.phi,
            runtime_ms=self.runtime_ms,
            n_calls=self.n_calls,
            method=self.method, device=self.device,
            detector=self.detector, anomaly_idx=self.anomaly_idx,
        )


# ============================================================
# Black-box predict function wrapper
# ============================================================

def get_predict_fn(detector, detector_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a function f(X) -> (n,) that returns the ANOMALY SCORE
    (higher = more anomalous) for each row of X (shape (n, d)).
    """
    if detector_name in ("IF", "LOF", "OCSVM"):
        # sklearn's score_samples returns *negative* for normal points
        return lambda X: -detector.model.score_samples(np.asarray(X, dtype=np.float64))

    if detector_name == "MLP":
        model = detector.model
        def _mlp_score(X):
            X = np.asarray(X, dtype=np.float32)
            X_hat = model.predict(X, verbose=0)
            return np.mean((X - X_hat) ** 2, axis=1).astype(np.float64)
        return _mlp_score

    raise ValueError(f"Unknown detector '{detector_name}'")


# ============================================================
# Cosine similarity helper
# ============================================================

def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query (d,) and each row of matrix (n, d)."""
    q = query / (np.linalg.norm(query) + 1e-12)
    M = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return M @ q

def compute_global_importance(
    predict_fn,
    X_normal: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Model-agnostic global feature importance bằng permutation sensitivity.

    Importance của feature j được đo bằng mức thay đổi trung bình
    của anomaly score khi feature j bị permutation trên tập normal.

    Parameters
    ----------
    predict_fn : callable
        Hàm nhận X -> anomaly score.
    X_normal : np.ndarray
        Training-normal samples ONLY.
    n_repeats : int
        Số lần permutation cho mỗi feature.
    seed : int
        Random seed.

    Returns
    -------
    importance : np.ndarray
        Vector importance đã normalize, tổng = 1.
    """
    X_normal = np.asarray(X_normal, dtype=np.float64)

    if X_normal.ndim != 2:
        raise ValueError(
            f"X_normal must be 2D, got shape={X_normal.shape}"
        )

    n_samples, n_features = X_normal.shape

    if n_samples < 2:
        raise ValueError(
            "Need at least 2 normal samples for permutation importance."
        )

    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")

    rng = np.random.default_rng(seed)

    # Baseline anomaly scores
    baseline_scores = np.asarray(
        predict_fn(X_normal),
        dtype=np.float64,
    ).reshape(-1)

    if len(baseline_scores) != n_samples:
        raise ValueError(
            "predict_fn(X_normal) returned an incompatible number "
            "of scores."
        )

    importance = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        deltas = []

        for _ in range(n_repeats):
            X_perm = X_normal.copy()

            permutation = rng.permutation(n_samples)
            X_perm[:, j] = X_perm[permutation, j]

            perm_scores = np.asarray(
                predict_fn(X_perm),
                dtype=np.float64,
            ).reshape(-1)

            delta = np.mean(
                np.abs(perm_scores - baseline_scores)
            )

            deltas.append(delta)

        importance[j] = np.mean(deltas)

    # Numerical safety
    importance = np.maximum(importance, 0.0)

    total = importance.sum()

    if total <= 1e-12:
        # Fallback nếu model không thể hiện sensitivity rõ ràng
        importance[:] = 1.0 / n_features
    else:
        importance /= total

    return importance

def compute_context_weights(
    global_importance: np.ndarray,
    feature_columns: list[str],
    context_columns: list[str],
) -> dict[str, float]:
    """
    Convert feature-level global importance into contextual weights.

    hour_sin + hour_cos are treated as one contextual variable: hour.
    power is excluded because it is an explanatory feature, not context.

    Returns
    -------
    dict[str, float]
        Normalized contextual weights, summing to 1.
    """
    global_importance = np.asarray(global_importance, dtype=np.float64)

    if len(global_importance) != len(feature_columns):
        raise ValueError(
            "global_importance and feature_columns must have "
            "the same length."
        )

    importance_map = dict(zip(feature_columns, global_importance))

    required_context = {
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "temperature",
        "humidity",
        "lux",
    }

    missing = required_context - set(context_columns)

    if missing:
        raise ValueError(
            f"Missing required context columns: {sorted(missing)}"
        )

    weights = {
        "hour": (
            importance_map["hour_sin"]
            + importance_map["hour_cos"]
        ),
        "day": importance_map["day_of_week"],
        "temperature": importance_map["temperature"],
        "humidity": importance_map["humidity"],
        "lux": importance_map["lux"],
    }

    total = sum(weights.values())

    if total <= 1e-12:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}

    return {
        key: value / total
        for key, value in weights.items()
    }

def select_context_pool(
    c_i,
    C_normal,
    normal_indices,
    frac=0.40,
    min_k=50,
    context_weights=None,
):
    """
    Select contextually relevant normal samples using Block-level Normalization
    (Áp dụng trên dữ liệu đầu vào C_normal và c_i đã qua MinMaxScaler [0, 1]).
    """
    c_i = np.asarray(c_i, dtype=float)
    C_normal = np.asarray(C_normal, dtype=float)
    normal_indices = np.asarray(normal_indices, dtype=int)
    n = len(normal_indices)

    if n == 0:
        raise ValueError("normal_indices is empty.")

    if context_weights is None:
        context_weights = {
            "hour": 0.25,
            "day": 0.15,
            "temperature": 0.2,
            "humidity": 0.2,
            "lux": 0.2,
        }

    # --------------------------------------------------------
    # 1. Hour: khôi phục về unit circle [-1, 1] từ scaled [0, 1]
    # x_raw = 2 * x_scaled - 1
    # --------------------------------------------------------
    sin_normal = 2.0 * C_normal[:, 0] - 1.0
    cos_normal = 2.0 * C_normal[:, 1] - 1.0
    sin_i = 2.0 * c_i[0] - 1.0
    cos_i = 2.0 * c_i[1] - 1.0

    # Chord distance trên unit circle [-1, 1], chia 2 để về [0, 1]
    d_hour = 0.5 * np.sqrt(
        (sin_normal - sin_i) ** 2 + (cos_normal - cos_i) ** 2
    )

    # --------------------------------------------------------
    # 2. Day: khôi phục về thang ngày gốc [0, 6] từ scaled [0, 1]
    # day_raw = day_scaled * 6.0
    # --------------------------------------------------------
    day_normal = C_normal[:, 2] * 6.0
    day_i = c_i[2] * 6.0

    day_diff = np.abs(day_normal - day_i)
    day_diff = np.minimum(day_diff, 7.0 - day_diff)
    d_day = day_diff / 3.5  # Max diff trong tuần là 3.5 ngày -> về [0, 1]

    # --------------------------------------------------------
    # 3. Environment: các biến môi trường vốn đã ở dải [0, 1]
    # --------------------------------------------------------
    d_temp = np.abs(C_normal[:, 3] - c_i[3])
    d_humid = np.abs(C_normal[:, 4] - c_i[4])
    d_lux = np.abs(C_normal[:, 5] - c_i[5])

    # --------------------------------------------------------
    # 4. Weighted Euclidean distance (tất cả thành phần đều chuẩn [0, 1])
    # --------------------------------------------------------
    dists_sq = (
        context_weights["hour"] * (d_hour ** 2)
        + context_weights["day"] * (d_day ** 2)
        + context_weights["temperature"] * (d_temp ** 2)
        + context_weights["humidity"] * (d_humid ** 2)
        + context_weights["lux"] * (d_lux ** 2)
    )
    dists = np.sqrt(dists_sq)

    # 5. Top-K nearest contextual normal samples
    k = min(max(min_k, int(np.ceil(frac * n))), n)
    order = np.argsort(dists, kind="stable")[:k]

    return normal_indices[order]

def inspect_context_pool(
    X: np.ndarray,
    idx: int,
    normal_idx: np.ndarray,
    context_idx: list[int],
    frac: float = 0.15,
    min_k: int = 50,
):
    """
    Inspect the context pool selected for one anomaly.
    Used for debugging/validation before running LIME-CA/SHAP-CA.
    """

    x_i = X[idx]
    c_i = x_i[context_idx]

    C_normal = X[normal_idx][:, context_idx]

    pool = select_context_pool(
        c_i=c_i,
        C_normal=C_normal,
        normal_indices=normal_idx,
        frac=frac,
        min_k=min_k,
    )

    print("\n" + "=" * 72)
    print("CONTEXT POOL INSPECTION")
    print("=" * 72)

    print(f"Anomaly index : {idx}")
    print(f"Normal pool   : {len(normal_idx)}")
    print(f"CA pool       : {len(pool)}")
    print(f"Context idx   : {context_idx}")

    print("\nAnomaly context:")
    for j in context_idx:
        print(f"  feature[{j}] = {x_i[j]:.4f}")

    print("\nSelected pool context mean:")
    for j in context_idx:
        print(
            f"  feature[{j}] = "
            f"{X[pool, j].mean():.4f}"
        )

    print("\nSelected pool context std:")
    for j in context_idx:
        print(
            f"  feature[{j}] = "
            f"{X[pool, j].std():.4f}"
        )

    return pool
# ============================================================
# 1. SHAP-CA : Context-aware KernelSHAP (Section 3.3, Alg. 1)
# ============================================================

def shap_ca(
    detector, detector_name: str,
    X: np.ndarray, idx: int, normal_idx: np.ndarray,
    context_idx: list[int],
    context_weights: dict | None = None,
    min_k: int = 50, frac: float = 0.4, M: int = 64, seed: int = 42,
) -> ExplanationResult:
    """KernelSHAP with context-aware background set B_i selected via select_context_pool."""
    import shap

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    predict_fn = get_predict_fn(detector, detector_name)
    x_i = X[idx].copy()
    c_i = x_i[context_idx]

    # ---- Step 1: select background set using shared pool ----
    C = X[normal_idx][:, context_idx]
    pool = select_context_pool(
        c_i,
        C,
        normal_idx,
        frac=frac,
        min_k=min_k,
        context_weights=context_weights,
    )
    if len(pool) >= min_k:
        bg_indices = rng.choice(pool, size=min_k, replace=False)
    else:
        bg_indices = pool

    B_i = X[bg_indices].copy()

    # ---- Step 2: KernelSHAP ----
    t0 = time.perf_counter()
    explainer = shap.KernelExplainer(predict_fn, B_i)
    sv = explainer.shap_values(x_i, nsamples=M, silent=True)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    phi = np.asarray(sv, dtype=np.float64).ravel()
    n_calls = M * 2

    return ExplanationResult(
        method="SHAP-CA", device="", detector=detector_name,
        anomaly_idx=idx, phi=phi, runtime_ms=runtime_ms, n_calls=n_calls,
        meta={"k": min_k, "frac": frac, "M": M, "n_background": len(bg_indices)},
    )

# ============================================================
# 2. LIME-std : Standard LIME (Section 3.5)
# ============================================================

def lime_std(
    detector, detector_name: str,
    X: np.ndarray, idx: int, normal_idx: np.ndarray,
    M: int = 150, sigma: float = 0.1,
    kernel_width: float = 0.75, ridge_alpha: float = 1.0,
    seed: int = 42,
) -> ExplanationResult:
    """
    Standard LIME: sample M points from an isotropic N(0, sigma^2) ball
    around x_i (sigma fixed, NOT context-dependent), fit a weighted,
    centered Ridge surrogate, return coef_.

    Surrogate form (matches LIME-CA / fidelity_mse exactly, so the two
    methods are comparable and fidelity is evaluated against the model
    that was actually fit):
        g(x) = phi . (x - x_i) + f(x_i)
    fit via Ridge(fit_intercept=False) on (Z - x_i, y - f(x_i)).
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    predict_fn = get_predict_fn(detector, detector_name)

    x_i = X[idx].copy()
    d = X.shape[1]

    # Gaussian perturbations around x_i (fixed, context-independent sigma)
    noise = rng.normal(0.0, sigma, size=(M, d))
    Z = x_i[None, :] + noise
    Z = np.clip(Z, 0.0, 1.0)

    t0 = time.perf_counter()
    f_i = predict_fn(x_i[None, :])[0]
    y = predict_fn(Z)

    distances = np.linalg.norm(Z - x_i[None, :], axis=1)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    if weights.max() < 1e-12:
        weights = np.ones_like(weights)
    else:
        weights = weights / (weights.max() + 1e-12)

    # Centered, no-intercept: forces g(x_i) = f(x_i) exactly, same
    # convention as LIME-CA and as fidelity_mse's surrogate reconstruction.
    Z_delta = Z - x_i[None, :]
    y_delta = y - f_i
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(Z_delta, y_delta, sample_weight=weights)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return ExplanationResult(
        method="LIME-std", device="", detector=detector_name,
        anomaly_idx=idx, phi=ridge.coef_.copy(), runtime_ms=runtime_ms,
        n_calls=M + 1,
        meta={
            "M": M, "sigma": sigma, "kernel_width": kernel_width,
            "sampling": "isotropic_gaussian_around_x_i",
            "regression": "centered_ridge_no_intercept",
        },
    )
# ============================================================
# 3. LIME-CA : Context-aware LIME 
# ============================================================
def lime_ca(
    detector, detector_name: str,
    X: np.ndarray, idx: int, normal_idx: np.ndarray,
    context_idx: list[int],
    context_weights: dict | None = None,
    frac: float = 0.4, min_k: int = 50,
    M: int = 150, sigma: float = 0.1,
    kernel_width: float = 0.75, ridge_alpha: float = 1.0,
    seed: int = 42,
) -> ExplanationResult:
    """
    LIME-CA chuẩn mực: Lấy 100% mẫu đối chứng từ pool bình thường cùng ngữ cảnh.
    Surrogate fit dạng centered không intercept: g(x) = phi . (x - x_i) + f(x_i).
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    predict_fn = get_predict_fn(detector, detector_name)

    x_i = X[idx].copy()
    d = X.shape[1]
    c_i = x_i[context_idx]

    # 1. Trích xuất context pool
    C = X[normal_idx][:, context_idx]
    pool = select_context_pool(
        c_i, C, normal_idx, frac=frac, min_k=min_k, context_weights=context_weights,
    )

    # 2. Lấy toàn bộ M mẫu từ pool bình thường thực tế
    base_indices = rng.choice(pool, size=M, replace=True)
    Z = X[base_indices].copy()

    # 3. Chỉ thêm rung lắc nhẹ vào các biến không phải context (ví dụ: power)
    context_set = set(context_idx)
    device_idx = [j for j in range(d) if j not in context_set]
    if device_idx:
        noise = rng.normal(0.0, sigma, size=(M, len(device_idx)))
        Z[:, device_idx] += noise
    Z = np.clip(Z, 0.0, 1.0)

    # 4. Tính khoảng cách và trọng số thích ứng
    t0 = time.perf_counter()
    f_i = float(predict_fn(x_i[None, :])[0])
    y = predict_fn(Z)

    distances = np.linalg.norm(Z - x_i[None, :], axis=1)
    
    # Kernel width thích ứng theo median khoảng cách thực tế để tránh triệt tiêu trọng số
    med_dist = float(np.median(distances))
    kw = max(kernel_width, med_dist) if med_dist > 1e-6 else kernel_width
    weights = np.exp(-(distances ** 2) / (kw ** 2))
    weights = weights / (weights.max() + 1e-12)

    # 5. Centered Ridge Regression (Không intercept)
    Z_delta = Z - x_i[None, :]
    y_delta = y - f_i
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(Z_delta, y_delta, sample_weight=weights)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return ExplanationResult(
        method="LIME-CA", device="", detector=detector_name,
        anomaly_idx=idx, phi=ridge.coef_.copy(), runtime_ms=runtime_ms,
        n_calls=M + 1,
        meta={"frac": frac, "M": M, "n_pool": len(pool)},
    )
# ============================================================
# 4. DeepSHAP (MLP only)
# ============================================================

def deepshap(
    detector, X: np.ndarray, idx: int, normal_idx: np.ndarray,
    context_idx: list[int],
    k: int = 50, frac: float = 0.4, seed: int = 42,
) -> ExplanationResult:
    """
    GradientExplainer explaining the ANOMALY SCORE (reconstruction MSE),
    not the raw reconstruction vector. Uses the same context-aware
    background as SHAP-CA.
    """
    import shap
    import tensorflow as tf
    from tensorflow import keras

    rng = np.random.default_rng(seed)
    ae_model = detector.model
    x_i = X[idx].astype(np.float32)
    c_i = x_i[context_idx]

# ---- Step 1: build context-aware background ----
    C = X[normal_idx][:, context_idx]
    pool = select_context_pool(c_i, C, normal_idx, frac=frac, min_k=k)
    if len(pool) >= k:
        bg_indices = rng.choice(pool, size=k, replace=False)
    else:
        bg_indices = pool
    B_i = X[bg_indices].astype(np.float32)
      # ---- Step 2: wrap AE into a scalar-anomaly-score model ----
    # Use Keras Functional API so the wrapper has .inputs / .outputs
    # attributes required by shap.GradientExplainer.
    # Note: tf.* ops cannot be applied to KerasTensors directly in TF 2.16+,
    # so we wrap the MSE computation in a Lambda layer.
    from tensorflow.keras import layers

    inp = keras.Input(shape=(X.shape[1],), dtype=tf.float32)
    x_hat = ae_model(inp, training=False)

    def _mse(x_in, x_out):
        return tf.reduce_mean((x_in - x_out) ** 2, axis=1, keepdims=True)

    score = layers.Lambda(
        lambda pair: _mse(pair[0], pair[1]),
        output_shape=(1,),
        name="anomaly_score",
    )([inp, x_hat])

    score_model = keras.Model(inputs=inp, outputs=score)
    # ---- Step 3: GradientExplainer ----
    t0 = time.perf_counter()
    explainer = shap.GradientExplainer(score_model, B_i)
    sv = explainer.shap_values(x_i[None, :])
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Normalize output shape to (d,)
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim == 3:       # (n_samples, n_features, n_outputs)
        phi = sv[0, :, 0]
    elif sv.ndim == 2:     # (n_samples, n_features)
        phi = sv[0]
    else:
        phi = sv.ravel()

    assert phi.shape == (X.shape[1],), (
        f"DeepSHAP expected phi shape ({X.shape[1]},), got {phi.shape}"
    )

    return ExplanationResult(
        method="DeepSHAP", device="", detector="MLP",
        anomaly_idx=idx, phi=phi, runtime_ms=runtime_ms,
        n_calls=2 * k,
        meta={"k": k, "frac": frac, "n_background": len(bg_indices)},
    )


# ============================================================
# 5. Integrated Gradients (MLP only)
# ============================================================

def integrated_gradients(
    detector, X: np.ndarray, idx: int,
    baseline: np.ndarray | None = None,
    steps: int = 50, seed: int = 42,
) -> ExplanationResult:
    """
    IG on the reconstruction-error score, with feature-wise median baseline.
    Deterministic; seed is accepted for interface uniformity.
    """
    import tensorflow as tf

    model = detector.model
    x_i = X[idx].astype(np.float32)
    if baseline is None:
        baseline = np.median(X, axis=0).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # Linear path: x_base -> x_i (steps+1 points, endpoints included)
    alphas = np.linspace(0.0, 1.0, steps + 1).astype(np.float32)
    path = baseline[None, :] + alphas[:, None] * (x_i - baseline)[None, :]

    t0 = time.perf_counter()
    path_t = tf.constant(path)
    with tf.GradientTape() as tape:
        tape.watch(path_t)
        recon = model(path_t, training=False)
        err = tf.reduce_mean((path_t - recon) ** 2, axis=1)
    grads = tape.gradient(err, path_t)                 # (steps+1, d)
    avg_grad = tf.reduce_mean(grads, axis=0).numpy()   # (d,)
    phi = (x_i - baseline) * avg_grad
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return ExplanationResult(
        method="IG", device="", detector="MLP",
        anomaly_idx=idx, phi=phi.astype(np.float64),
        runtime_ms=runtime_ms, n_calls=steps + 1,
        meta={"steps": steps},
    )


# ============================================================
# Unified interface
# ============================================================

_METHODS = {
    "SHAP-CA":  shap_ca,
    "LIME-std": lime_std,
    "LIME-CA":  lime_ca,
    "DeepSHAP": deepshap,
    "IG":       integrated_gradients,
}

def explain_anomaly(
    detector, detector_name: str,
    X: np.ndarray, idx: int, normal_idx: np.ndarray,
    method: str, device: str = "",
    context_idx: list[int] | None = None,
    **kwargs,
) -> ExplanationResult:
    """Dispatch to the selected explanation method."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Available: {list(_METHODS)}")

    fn = _METHODS[method]

    if method in ("SHAP-CA", "LIME-CA", "DeepSHAP"):
        if context_idx is None:
            raise ValueError(f"'{method}' requires context_idx.")
        kwargs["context_idx"] = context_idx

    if method == "LIME-std":
        result = fn(detector, detector_name, X, idx, normal_idx, **kwargs)
    elif method == "IG":
        result = fn(detector, X, idx, **kwargs)
    elif method == "DeepSHAP":
        result = fn(detector, X, idx, normal_idx, **kwargs)
    else:
        result = fn(detector, detector_name, X, idx, normal_idx, **kwargs)

    result.device = device
    return result
# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_prep_cubems import load_zone_dataframe, FEATURE_COLUMNS, CONTEXT_COLUMNS
    from anomaly_detection import (
        build_detectors, threshold_95, get_top_anomalies, get_normal_indices,
    )

    print("=" * 72)
    print("SMOKE TEST — explainers.py")
    print("=" * 72)

    zone = "floor2_z1"
    df = load_zone_dataframe(zone, verbose=False)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    context_idx = [FEATURE_COLUMNS.index(c) for c in CONTEXT_COLUMNS]

    print(f"\nZone: {zone}  |  X shape: {X.shape}")
    print(f"Context indices: {context_idx} -> "
          f"{[FEATURE_COLUMNS[i] for i in context_idx]}")

    dets = build_detectors(X, ["IF", "MLP"])
    print(f"Detectors: {list(dets)}")

    for det_name, det in dets.items():
        scores = det.score(X)
        theta = threshold_95(scores)
        normal_idx = get_normal_indices(scores, theta)
        top20 = get_top_anomalies(scores, k=20)
        target = int(top20[0])  # idx 24 should be top-1

        print(f"\n{'-'*72}")
        print(f"Detector: {det_name}  |  target anomaly idx = {target}")
        print(f"{'-'*72}")

        methods = ["SHAP-CA", "LIME-std", "LIME-CA"]
        if det_name == "MLP":
            methods += ["DeepSHAP", "IG"]

        for m in methods:
            res = explain_anomaly(
                det, det_name, X, target, normal_idx,
                m, device=zone,
                context_idx=context_idx,
            )
            
            print(f"  [{m:9s}] phi.shape={res.phi.shape}  "
                  f"||phi||_1={np.abs(res.phi).sum():.4f}  "
                  f"runtime={res.runtime_ms:7.2f} ms  "
                  f"n_calls={res.n_calls}")

    # Reproducibility check for LIME-CA (has randomness)
    print(f"\n{'-'*72}")
    print("REPRODUCIBILITY CHECK — LIME-CA")
    print(f"{'-'*72}")
    det = dets["IF"]
    scores = det.score(X)
    theta = threshold_95(scores)
    normal_idx = get_normal_indices(scores, theta)
    target = int(get_top_anomalies(scores, k=20)[0])

    # --------------------------------------------------------------
    # DEBUG: inspect context-aware normal pool
    # --------------------------------------------------------------
    pool = inspect_context_pool(
        X=X,
        idx=target,
        normal_idx=normal_idx,
        context_idx=context_idx,
        frac=0.15,
        min_k=50,
        )

    print("\nSelected pool indices:")
    print(pool[:20])
    r1 = explain_anomaly(det, "IF", X, target, normal_idx,
                         "LIME-CA", context_idx=context_idx, seed=42)
    r2 = explain_anomaly(det, "IF", X, target, normal_idx,
                         "LIME-CA", context_idx=context_idx, seed=42)
    diff = float(np.max(np.abs(r1.phi - r2.phi)))
    print(f"  max|phi_1 - phi_2| (same seed) = {diff:.2e}")
    print(f"  Status: {'OK' if diff < 1e-10 else 'FAIL'}")
    r3 = explain_anomaly(det, "IF", X, target, normal_idx,
                         "LIME-CA", context_idx=context_idx, seed=43)
    diff3 = float(np.max(np.abs(r1.phi - r3.phi)))
    print(f"  max|phi_1 - phi_3| (diff seed) = {diff3:.2e}")
    print(f"  (expect > 0 for LIME-CA due to random base sampling)")

    print("\n[DONE] Smoke test complete.")