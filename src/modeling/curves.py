"""Single-split train/validation loss curves for non-iterative models.

Decision trees and bagging ensembles have no boosting iterations, so a genuine
train-vs-validation loss curve is produced by sweeping a complexity parameter
(tree depth, forest size) and recording log loss on a fixed train/valid split.
Preprocessing is fit on the train rows only.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.preprocess import build_preprocess_pipeline


def _prep(X_tr: pd.DataFrame, X_va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = build_preprocess_pipeline()
    pre.fit(X_tr)
    return pre.transform(X_tr), pre.transform(X_va)


def _ll(y: pd.Series, proba: np.ndarray) -> float:
    return float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)))


def param_loss_curve(
    estimator_factory: Callable[[Any], Any],
    values: Sequence[Any],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
) -> tuple[list[float], list[float]]:
    """Fit a fresh estimator per parameter value; return train/valid log loss.

    `estimator_factory(value)` must return an unfitted estimator configured with
    that parameter value (e.g. ``lambda d: DecisionTreeClassifier(max_depth=d)``).
    """
    A, B = _prep(X_tr, X_va)
    train_loss, valid_loss = [], []
    for v in values:
        model = estimator_factory(v)
        model.fit(A, y_tr)
        train_loss.append(_ll(y_tr, model.predict_proba(A)[:, 1]))
        valid_loss.append(_ll(y_va, model.predict_proba(B)[:, 1]))
    return train_loss, valid_loss


def warm_start_loss_curve(
    estimator: Any,
    n_values: Sequence[int],
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
) -> tuple[list[float], list[float]]:
    """Grow a warm-start forest and record train/valid log loss by tree count.

    Works for RandomForest / ExtraTrees / GradientBoosting (anything supporting
    ``warm_start`` + ``n_estimators``). `n_values` must be increasing.
    """
    A, B = _prep(X_tr, X_va)
    estimator.set_params(warm_start=True)
    train_loss, valid_loss = [], []
    for n in n_values:
        estimator.set_params(n_estimators=int(n))
        estimator.fit(A, y_tr)
        train_loss.append(_ll(y_tr, estimator.predict_proba(A)[:, 1]))
        valid_loss.append(_ll(y_va, estimator.predict_proba(B)[:, 1]))
    return train_loss, valid_loss
