"""Label propagation helpers for semi-supervised learning."""

import os
from typing import Tuple
import numpy as np
from sklearn.semi_supervised import LabelPropagation


def load_embeddings(embeddings_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and labels saved by the extraction step.

    Expects files `features.npy` and `labels.npy` in `embeddings_dir`.
    """
    X = np.load(os.path.join(embeddings_dir, "features.npy"))
    y = np.load(os.path.join(embeddings_dir, "labels.npy"))
    return X, y


def mask_labels(y: np.ndarray, unlabeled_ratio: float = 0.5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return a copy of y with a fraction set to -1 (unlabeled) and the boolean mask used.

    Args:
        y: ground-truth labels array of shape (N,)
        unlabeled_ratio: fraction in (0,1] to mark as unlabeled (-1)
        seed: RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(y.shape[0]) < unlabeled_ratio
    y_missing = y.copy()
    y_missing[mask] = -1
    return y_missing, mask


def run_label_propagation(
    X: np.ndarray,
    y_with_missing: np.ndarray,
    kernel: str = "knn",
    n_neighbors: int = 5,
    gamma: float | None = None,
):
    """Fit sklearn LabelPropagation and return propagated labels and the model.

    Supports `kernel` in {"knn", "rbf"}.
    - For `rbf`, if `gamma` is None, defaults to 20.0.
    - For `knn`, uses `n_neighbors`.
    """
    kernel = kernel.lower()
    if kernel not in {"knn", "rbf"}:
        raise ValueError("kernel must be 'knn' or 'rbf'")

    if kernel == "rbf":
        model = LabelPropagation(kernel="rbf", gamma=20.0 if gamma is None else gamma)
    else:
        model = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)

    model.fit(X, y_with_missing)
    return model.transduction_, model


def save_numpy(dirpath: str, **arrays) -> None:
    """Save named numpy arrays into `dirpath` as `<name>.npy`. Creates the dir if needed."""
    os.makedirs(dirpath, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(dirpath, f"{name}.npy"), arr)
