"""Tuned champion configuration per model family.

These hyperparameters are the GridSearchCV winners from notebooks 06-09
(selected on the stratified search sample). Keeping them in one place gives a
single source of truth for the comparison notebook and the Stage-6 pipeline
port. Heavy optional dependencies (xgboost/lightgbm/catboost) are imported
lazily so importing this module never fails.
"""

from __future__ import annotations

from typing import Any, Callable

# CV AUC achieved on the search sample (for reference / ordering only).
SEARCH_CV_AUC: dict[str, float] = {
    "logistic_regression": 0.7271,
    "decision_tree": 0.7070,
    "random_forest": 0.7581,
    "xgboost": 0.7626,
    "lightgbm": 0.7615,
    "catboost": 0.7616,
}

FAMILY: dict[str, str] = {
    "logistic_regression": "linear",
    "decision_tree": "tree",
    "random_forest": "ensemble",
    "xgboost": "boosting",
    "lightgbm": "boosting",
    "catboost": "boosting",
}


def _logistic_regression(pos_weight: float) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=1.0,
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )


def _decision_tree(pos_weight: float) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        criterion="entropy",
        max_depth=8,
        min_samples_leaf=200,
        class_weight="balanced",
        random_state=42,
    )


def _random_forest(pos_weight: float) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def _xgboost(pos_weight: float) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def _lightgbm(pos_weight: float) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _catboost(pos_weight: float) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=7.0,
        scale_pos_weight=pos_weight,
        loss_function="Logloss",
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )


_FACTORIES: dict[str, Callable[[float], Any]] = {
    "logistic_regression": _logistic_regression,
    "decision_tree": _decision_tree,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "catboost": _catboost,
}


def champion_names() -> list[str]:
    """Names of all tuned champions, ordered by search CV AUC (desc)."""
    return sorted(_FACTORIES, key=lambda n: SEARCH_CV_AUC.get(n, 0.0), reverse=True)


def build_champion(name: str, pos_weight: float) -> Any:
    """Return the unfitted tuned estimator for one family champion."""
    if name not in _FACTORIES:
        raise KeyError(f"Unknown champion '{name}'. Options: {list(_FACTORIES)}")
    return _FACTORIES[name](pos_weight)


def build_all_champions(pos_weight: float) -> dict[str, Any]:
    """Return unfitted tuned estimators for every family champion."""
    return {name: _FACTORIES[name](pos_weight) for name in champion_names()}
