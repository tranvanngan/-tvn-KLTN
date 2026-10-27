"""
anomaly_detection.py
====================
Fit four unsupervised anomaly detectors on the preprocessed 6-D features
(see data_prep.py) and produce per-sample anomaly scores.

Detectors (matching paper Section 4.2):
  1. Isolation Forest     (IF)    -- contamination=0.05, n_estimators=100
  2. Local Outlier Factor (LOF)   -- n_neighbors=20, novelty=True
  3. One-Class SVM        (OCSVM) -- RBF kernel, nu=0.05, gamma='auto'
  4. MLP Autoencoder      (MLP)   -- 6->16->8->16->6, reconstruction error

Reproducibility contract
------------------------
- All RNG seeds set (Python, NumPy, scikit-learn, TensorFlow).
- TF deterministic ops enabled via env vars (set BEFORE TF import).
- Running twice with the same --seed MUST produce bit-identical scores
  (up to 1e-6 tolerance for MLP, 1e-12 for sklearn methods).
- Script self-verifies reproducibility unless --no-verify is passed.

Design note (baseline reproduction)
-----------------------------------
v1 trained all detectors on the full 361 samples and selected the top-20
highest-scoring samples as "anomalies". This design is retained for
baseline reproduction; the methodological limitation (no train/test
split) is discussed in Section 7.1 of the paper. A future extension
will add train/test separation.

Outputs (saved to results/)
---------------------------
For each (device, detector) pair:
    results/{device}_{detector}_scores.npy          (361,) float64
    results/{device}_{detector}_normal_indices.npy  int64
    results/{device}_{detector}_theta.json          {"theta": float}
    results/{device}_{detector}_top20.json          {"indices": [...], "scores": [...]}
"""
from __future__ import annotations

# --- Set deterministic env vars BEFORE importing tensorflow ---
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make `data_prep` importable when script is run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep_cubems import load_zone_dataframe as load_device_dataframe
from data_prep_cubems import FEATURE_COLUMNS

# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            # TF >= 2.9
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
        try:
            # TF >= 2.7
            tf.keras.utils.set_random_seed(seed)
        except AttributeError:
            pass
    except ImportError:
        pass


# ============================================================
# Detector interface
# ============================================================

@dataclass
class _SklearnDetector:
    """Wrapper around sklearn detectors.

    sklearn's `score_samples` returns *negative* values for normal points
    (higher = more normal). We negate the output so that higher = more
    anomalous, matching the MLP reconstruction-error convention.
    """
    name: str
    model: object

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


@dataclass
class _MLPDetector:
    """Wrapper around the Keras autoencoder (anomaly = reconstruction error)."""
    name: str
    model: object  # keras.Model

    def score(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.model.predict(X, verbose=0)
        return np.mean((X - X_hat) ** 2, axis=1)


# ============================================================
# Detector factory
# ============================================================

def _fit_mlp_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    seed: int = 42,
    max_epochs: int = 100,
    patience: int = 10,
    verbose: int = 0,
) -> _MLPDetector:
    """Build and fit the MLP autoencoder defined in the paper with validation monitoring."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(seed)

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2, seed=seed),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(X_train.shape[1], activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse")

    # Nếu có X_val thì monitor val_loss, nếu không thì fallback về loss
    monitor_metric = "val_loss" if X_val is not None else "loss"
    val_data = (X_val, X_val) if X_val is not None else None

    es = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=patience,
        restore_best_weights=True,
    )
    model.fit(
        X_train, X_train,
        epochs=max_epochs,
        batch_size=32,
        shuffle=True,
        validation_data=val_data,
        callbacks=[es],
        verbose=verbose,
    )
    return _MLPDetector("MLP", model)

def build_detectors(
    X_train: np.ndarray,
    detector_names: list[str],
    X_val: np.ndarray | None = None,
    random_state: int = 42,
) -> dict:
    """Fit the requested detectors on X_train. Returns {name: detector}."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    set_global_seed(random_state)
    detectors: dict = {}

    if "IF" in detector_names:
        if_ = IsolationForest(
            contamination=0.05,
            random_state=random_state,
            n_estimators=100,
        )
        if_.fit(X_train)
        detectors["IF"] = _SklearnDetector("IF", if_)

    if "LOF" in detector_names:
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True,
        )
        lof.fit(X_train)
        detectors["LOF"] = _SklearnDetector("LOF", lof)

    if "OCSVM" in detector_names:
        oc = OneClassSVM(kernel="rbf", nu=0.05, gamma="auto")
        oc.fit(X_train)
        detectors["OCSVM"] = _SklearnDetector("OCSVM", oc)

    if "MLP" in detector_names:
        try:
            detectors["MLP"] = _fit_mlp_autoencoder(
                X_train, X_val=X_val, seed=random_state
            )
        except ImportError:
            print("  [WARN] TensorFlow not available; skipping MLP.")
    return detectors

# Anomaly selection

def threshold_95(scores: np.ndarray) -> float:
    """Paper: theta = 95th percentile of anomaly scores."""
    return float(np.percentile(scores, 95))


def get_top_anomalies(scores: np.ndarray, k: int = 20) -> np.ndarray:
    """Indices of the k highest-scoring samples (descending by score)."""
    return np.argsort(scores)[-k:][::-1]


def get_normal_indices(scores: np.ndarray, theta: float) -> np.ndarray:
    """Indices with score <= theta, used as 'normal pool' by SHAP-CA / LIME-CA."""
    return np.where(scores <= theta)[0]

# Result container

@dataclass
class DetectionResult:
    device: str
    detector: str
    scores: np.ndarray
    theta: float
    top20_indices: np.ndarray
    top20_scores: np.ndarray
    normal_indices: np.ndarray

    @classmethod
    def from_scores(cls, device: str, detector: str,
                    scores: np.ndarray) -> "DetectionResult":
        theta = threshold_95(scores)
        top20 = get_top_anomalies(scores, k=20)
        normal = get_normal_indices(scores, theta)
        return cls(
            device=device, detector=detector, scores=scores,
            theta=theta, top20_indices=top20,
            top20_scores=scores[top20], normal_indices=normal,
        )

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.device}_{self.detector}"

        np.save(output_dir / f"{stem}_scores.npy", self.scores)
        np.save(output_dir / f"{stem}_normal_indices.npy", self.normal_indices)

        with open(output_dir / f"{stem}_theta.json", "w") as f:
            json.dump({"theta": self.theta}, f, indent=2)

        with open(output_dir / f"{stem}_top20.json", "w") as f:
            json.dump({
                "indices": self.top20_indices.tolist(),
                "scores": self.top20_scores.tolist(),
            }, f, indent=2)

    @classmethod
    def load(cls, output_dir: Path, device: str,
             detector: str) -> "DetectionResult":
        output_dir = Path(output_dir)
        stem = f"{device}_{detector}"
        with open(output_dir / f"{stem}_theta.json") as f:
            theta = json.load(f)["theta"]
        with open(output_dir / f"{stem}_top20.json") as f:
            data = json.load(f)
        return cls(
            device=device, detector=detector,
            scores=np.load(output_dir / f"{stem}_scores.npy"),
            theta=theta,
            top20_indices=np.array(data["indices"]),
            top20_scores=np.array(data["scores"]),
            normal_indices=np.load(output_dir / f"{stem}_normal_indices.npy"),
        )


# Sanity checks

def check_fridge_zero_power(df, top20_indices) -> None:
    """F4: Fridge has one sample with raw_power == 0.0 W. Is it flagged?"""
    print("\n  [CHECK F4] Fridge raw_power == 0.0 W")
    zero_idx = np.where(df["raw_power"].values == 0.0)[0]
    if len(zero_idx) == 0:
        print("    -> Khong tim thay mau power = 0 W.")
        return
    for idx in zero_idx:
        status = "CO trong top-20" if idx in top20_indices else "KHONG trong top-20"
        print(f"    -> idx {idx}: {status}")


def check_printer_leading_gap(top20_indices) -> None:
    """F2: Printer leading gap idx 0-3 was backward-filled. Any in top-20?"""
    print("\n  [CHECK F2] Printer leading gap idx 0-3 (imputed)")
    imputed = set(range(4))
    hits = [int(i) for i in top20_indices if i in imputed]
    if hits:
        print(f"    -> CO {len(hits)} imputed sample(s) in top-20: {hits}")
        print("    -> Can exclude khoi analysis (xem FINDINGS.md F2).")
    else:
        print("    -> Khong co imputed sample nao trong top-20. OK.")

# Main pipeline per device

def run_device(
    device: str,
    detector_names: list[str],
    output_dir: Path,
    seed: int = 42,
    save: bool = True,
) -> dict:
    print(f"\n{'='*72}")
    print(f"DEVICE: {device.upper()}")
    print(f"{'='*72}")

    df = load_device_dataframe(device, verbose=False)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    print(f"  Loaded: {X.shape[0]} samples x {X.shape[1]} features")

    detectors = build_detectors(X, detector_names, random_state=seed)

    results: dict = {}
    for name, det in detectors.items():
        scores = det.score(X)
        res = DetectionResult.from_scores(device, name, scores)
        results[name] = res

        print(f"\n  [{name}]")
        print(f"    score range  : {scores.min():.6f} .. {scores.max():.6f}")
        print(f"    theta (95%)  : {res.theta:.6f}")
        print(f"    n normal     : {len(res.normal_indices)}")
        print(f"    top-5 idx    : {res.top20_indices[:5].tolist()}")
        print(f"    top-5 scores : {[f'{s:.4f}' for s in res.top20_scores[:5]]}")

        if save:
            res.save(output_dir)

    # Cross-reference sanity checks (findings F2, F4)
    if device == "fridge" and "IF" in results:
        check_fridge_zero_power(df, results["IF"].top20_indices)
    if device == "printer" and "IF" in results:
        check_printer_leading_gap(results["IF"].top20_indices)

    return results

# Reproducibility verification

def verify_reproducibility(
    device: str,
    detector_names: list[str],
    seed: int = 42,
) -> bool:
    """Fit detectors twice with the same seed; compare scores element-wise."""
    print(f"\n{'='*72}")
    print("REPRODUCIBILITY VERIFICATION")
    print(f"{'='*72}")

    df = load_device_dataframe(device, verbose=False)
    X = df[FEATURE_COLUMNS].values.astype(np.float64)

    dets_a = build_detectors(X, detector_names, random_state=seed)
    dets_b = build_detectors(X, detector_names, random_state=seed)

    all_ok = True
    for name in dets_a:
        sa = dets_a[name].score(X)
        sb = dets_b[name].score(X)
        # MLP may have tiny float differences; sklearn must be bit-identical
        atol = 1e-6 if name == "MLP" else 1e-12
        ok = np.allclose(sa, sb, atol=atol, rtol=0)
        max_diff = float(np.max(np.abs(sa - sb)))
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name:6s}  max|delta| = {max_diff:.2e}  (atol={atol:.0e})")
        all_ok = all_ok and ok

    return all_ok

# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Fit anomaly detectors on CLEF preprocessed data.",
    )
    p.add_argument("--devices", nargs="+", default=["fridge", "printer"],
                   choices=["fridge", "printer"])
    p.add_argument("--detectors", nargs="+",
                   default=["IF", "LOF", "OCSVM", "MLP"],
                   choices=["IF", "LOF", "OCSVM", "MLP"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--no-mlp", action="store_true",
                   help="Skip MLP (faster; no TensorFlow required)")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip reproducibility verification")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save results to disk")
    return p.parse_args()


def main():
    args = parse_args()

    detectors = list(args.detectors)
    if args.no_mlp and "MLP" in detectors:
        detectors.remove("MLP")

    set_global_seed(args.seed)
    print(f"Seed:        {args.seed}")
    print(f"Devices:     {args.devices}")
    print(f"Detectors:   {detectors}")
    print(f"Output dir:  {args.output_dir}")

    for device in args.devices:
        run_device(
            device=device,
            detector_names=detectors,
            output_dir=args.output_dir,
            seed=args.seed,
            save=not args.no_save,
        )

    if not args.no_verify:
        ok = verify_reproducibility(
            device=args.devices[0],
            detector_names=detectors,
            seed=args.seed,
        )
        if not ok:
            print("\n[ERROR] Reproducibility check FAILED. Investigate before proceeding.")
            sys.exit(1)
        print("\n[OK] Reproducibility verified.")


if __name__ == "__main__":
    main()