"""Epoch-wise loss curve for linear models via incremental SGD training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.modeling.boosting import LossCurve
from src.preprocess import build_preprocess_pipeline


def fit_logreg_sgd_curve(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any] | None = None,
    n_epochs: int = 60,
) -> LossCurve:
    """Logistic regression trained with SGD; train/valid log loss per epoch.

    Logistic regression has no native iteration history, so we fit an
    equivalent log-loss SGD classifier incrementally with `partial_fit` and
    record the loss after each epoch. Preprocessing is fit on train rows only,
    followed by standardisation (required for well-behaved SGD).
    """
    base = {
        "loss": "log_loss",
        "penalty": "l2",
        "alpha": 1e-4,
        "learning_rate": "optimal",
        "class_weight": "balanced",
        "random_state": 42,
    }
    base.update(params or {})

    # partial_fit does not accept class_weight='balanced'; resolve to a dict.
    if base.get("class_weight") == "balanced":
        classes_arr = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes_arr, y=y_tr.to_numpy())
        base["class_weight"] = {0: float(weights[0]), 1: float(weights[1])}

    pre = build_preprocess_pipeline()
    pre.fit(X_tr)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(pre.transform(X_tr))
    Xva = scaler.transform(pre.transform(X_va))

    model = SGDClassifier(**base)
    classes = np.array([0, 1])
    train_loss, valid_loss = [], []
    rng = np.random.default_rng(42)
    n = Xtr.shape[0]
    for _ in range(n_epochs):
        perm = rng.permutation(n)
        model.partial_fit(Xtr[perm], y_tr.to_numpy()[perm], classes=classes)
        p_tr = _proba(model, Xtr)
        p_va = _proba(model, Xva)
        train_loss.append(log_loss(y_tr, p_tr))
        valid_loss.append(log_loss(y_va, p_va))
    return LossCurve(
        model=model,
        train_loss=train_loss,
        valid_loss=valid_loss,
        best_iteration=int(np.argmin(valid_loss)),
        extra={"scaler": scaler, "preprocess": pre},
    )


def _proba(model: SGDClassifier, X: np.ndarray) -> np.ndarray:
    """Clip decision-function-based probabilities for stable log loss."""
    scores = model.decision_function(X)
    proba = 1.0 / (1.0 + np.exp(-scores))
    return np.clip(proba, 1e-6, 1 - 1e-6)
