"""Loss-curve fitters for gradient-boosting libraries.

Each fitter trains on a single (train, valid) split with preprocessing fit on
the train rows only, records per-iteration train/validation log loss, and
returns the fitted model plus the two loss arrays for `plot_loss_curve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.preprocess import build_preprocess_pipeline


@dataclass
class LossCurve:
    """Fitted model and its per-iteration train/validation loss."""

    model: Any
    train_loss: list[float]
    valid_loss: list[float]
    best_iteration: int
    metric: str = "log_loss"
    extra: dict[str, Any] = field(default_factory=dict)


def _prep_split(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit preprocess on train rows only and transform both splits."""
    pre = build_preprocess_pipeline()
    pre.fit(X_tr)
    return pre.transform(X_tr), pre.transform(X_va)


def fit_xgb_curve(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
) -> LossCurve:
    """XGBoost with logloss eval on train/valid; native evals_result."""
    from xgboost import XGBClassifier

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    base = {
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "scale_pos_weight": pos_weight,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    base.update(params or {})
    Xtr, Xva = _prep_split(X_tr, X_va)
    model = XGBClassifier(**base)
    model.fit(Xtr, y_tr, eval_set=[(Xtr, y_tr), (Xva, y_va)], verbose=False)
    res = model.evals_result()
    train_loss = res["validation_0"]["logloss"]
    valid_loss = res["validation_1"]["logloss"]
    return LossCurve(
        model=model,
        train_loss=list(train_loss),
        valid_loss=list(valid_loss),
        best_iteration=int(np.argmin(valid_loss)),
    )


def fit_lgbm_curve(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
) -> LossCurve:
    """LightGBM with binary_logloss recorded on train/valid."""
    import lightgbm as lgb

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    base = {
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "scale_pos_weight": pos_weight,
        "n_jobs": -1,
        "random_state": 42,
    }
    base.update(params or {})
    Xtr, Xva = _prep_split(X_tr, X_va)
    evals: dict[str, dict[str, list]] = {}
    model = lgb.LGBMClassifier(**base)
    model.fit(
        Xtr,
        y_tr,
        eval_set=[(Xtr, y_tr), (Xva, y_va)],
        eval_names=["train", "valid"],
        eval_metric="binary_logloss",
        callbacks=[lgb.record_evaluation(evals), lgb.log_evaluation(0)],
    )
    train_loss = evals["train"]["binary_logloss"]
    valid_loss = evals["valid"]["binary_logloss"]
    return LossCurve(
        model=model,
        train_loss=list(train_loss),
        valid_loss=list(valid_loss),
        best_iteration=int(np.argmin(valid_loss)),
    )


def fit_catboost_curve(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 600,
) -> LossCurve:
    """CatBoost with Logloss eval on train/valid via get_evals_result."""
    from catboost import CatBoostClassifier, Pool

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    base = {
        "iterations": num_boost_round,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 3.0,
        "scale_pos_weight": pos_weight,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }
    base.update(params or {})
    Xtr, Xva = _prep_split(X_tr, X_va)
    model = CatBoostClassifier(**base)
    model.fit(Pool(Xtr, y_tr), eval_set=Pool(Xva, y_va), verbose=False)
    res = model.get_evals_result()
    train_loss = res["learn"]["Logloss"]
    valid_loss = res["validation"]["Logloss"]
    return LossCurve(
        model=model,
        train_loss=list(train_loss),
        valid_loss=list(valid_loss),
        best_iteration=int(np.argmin(valid_loss)),
    )


def fit_gbm_curve(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 400,
) -> LossCurve:
    """sklearn GradientBoostingClassifier; staged log loss over boosting stages."""
    from sklearn.ensemble import GradientBoostingClassifier

    base = {
        "n_estimators": num_boost_round,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "random_state": 42,
    }
    base.update(params or {})
    Xtr, Xva = _prep_split(X_tr, X_va)
    model = GradientBoostingClassifier(**base)
    model.fit(Xtr, y_tr)

    train_loss, valid_loss = [], []
    for p_tr in model.staged_predict_proba(Xtr):
        train_loss.append(log_loss(y_tr, np.clip(p_tr[:, 1], 1e-6, 1 - 1e-6)))
    for p_va in model.staged_predict_proba(Xva):
        valid_loss.append(log_loss(y_va, np.clip(p_va[:, 1], 1e-6, 1 - 1e-6)))
    return LossCurve(
        model=model,
        train_loss=train_loss,
        valid_loss=valid_loss,
        best_iteration=int(np.argmin(valid_loss)),
    )
