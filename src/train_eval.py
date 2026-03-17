"""Training and evaluation helpers for a baseline classifier."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from .thresholds import sweep_thresholds, choose_best_threshold


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Stratified split + StandardScaler + LogisticRegression(class_weight='balanced').
    Returns fitted pipeline + X_test + y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def evaluate_classification(
    clf: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, Optional[float]]:
    """
    Returns (classification_report_dict, y_pred, auc_or_None)
    """
    y_pred = clf.predict(X_test)
    report_dict = classification_report(y_test, y_pred, digits=2, output_dict=True)

    auc = None
    if hasattr(clf, "predict_proba") and len(np.unique(y_test)) == 2:
        proba = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))

    return report_dict, y_pred, auc
