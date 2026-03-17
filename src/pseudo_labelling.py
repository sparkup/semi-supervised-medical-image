"""Pseudo-labelling utilities with confidence thresholding."""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def run_pseudo_labelling(X: np.ndarray, y_seed: np.ndarray, confidence_threshold: float = 0.9, max_iters: int = 3, class_weight: str | None = "balanced", random_state: int = 42) -> np.ndarray:
    """
    Iterative pseudo-labelling with confidence thresholding.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y_seed (np.ndarray): Labels with -1 for unlabeled samples
        confidence_threshold (float): Minimum probability to accept pseudo-label
        max_iters (int): Number of iterations
        class_weight (str|None): Class weighting for LogisticRegression (default "balanced")
        random_state (int): Random seed for reproducibility

    Returns:
        np.ndarray: Updated labels array after pseudo-labelling
    """
    y = y_seed.copy()

    for _ in range(max_iters):
        labeled_idx = y != -1
        if labeled_idx.sum() == 0:
            break

        clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=random_state)
        clf.fit(X[labeled_idx], y[labeled_idx])

        probs = clf.predict_proba(X)
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        pick = (y == -1) & (conf >= confidence_threshold)
        if pick.sum() == 0:
            break

        y[pick] = preds[pick]

    return y

def save_pseudo_labels(output_dir: str, labels: np.ndarray) -> None:
    """
    Save pseudo-labels to a given directory.

    Args:
        output_dir (str): Path to directory where labels will be saved
        labels (np.ndarray): Array of labels to save
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "labels_pseudo.npy"), labels)
