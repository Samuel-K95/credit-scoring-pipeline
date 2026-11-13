"""Pipeline construction and GridSearchCV helpers shared by notebooks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.modeling.data import ModelingData
from src.modeling.metrics import classification_metrics
from src.preprocess import build_preprocess_pipeline

MODEL_STEP = "model"
PREPROCESS_STEP = "preprocess"

# Scoring objectives used across notebooks (AUC drives selection on imbalanced PD).
DEFAULT_SCORING: dict[str, str] = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "neg_log_loss": "neg_log_loss",
}


def make_pipeline(estimator: Any, with_preprocess: bool = True) -> Pipeline:
    """Wrap an estimator so preprocessing is refit inside each CV fold.

    Keeping preprocessing inside the pipeline guarantees impute/cap statistics
    come only from each fold's training rows (no leakage into validation).
    """
    steps = []
    if with_preprocess:
        steps.append((PREPROCESS_STEP, build_preprocess_pipeline()))
    steps.append((MODEL_STEP, estimator))
    return Pipeline(steps)


def prefix_grid(param_grid: dict[str, list]) -> dict[str, list]:
    """Prefix bare parameter names with the pipeline model-step name."""
    return {f"{MODEL_STEP}__{k}": v for k, v in param_grid.items()}


def run_grid_search(
    estimator: Any,
    param_grid: dict[str, list],
    data: ModelingData,
    *,
    scoring: dict[str, str] | str = DEFAULT_SCORING,
    refit: str = "roc_auc",
    with_preprocess: bool = True,
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Run GridSearchCV over the saved stratified folds and report timing."""
    pipe = make_pipeline(estimator, with_preprocess=with_preprocess)
    grid = prefix_grid(param_grid)
    gs = GridSearchCV(
        pipe,
        param_grid=grid,
        scoring=scoring,
        refit=refit,
        cv=data.cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    n_combos = int(np.prod([len(v) for v in param_grid.values()])) if param_grid else 1
    print(f"Fitting {n_combos} candidate(s) x {len(data.cv)} folds ...", flush=True)
    t0 = time.perf_counter()
    gs.fit(data.X_train, data.y_train)
    dt = time.perf_counter() - t0
    best = refit if isinstance(scoring, dict) else "score"
    print(f"Done in {dt:.1f}s. Best {best}={gs.best_score_:.4f}", flush=True)
    print("Best params:", gs.best_params_, flush=True)
    return gs


def cv_results_frame(gs: GridSearchCV, refit: str = "roc_auc") -> pd.DataFrame:
    """Tidy the most useful GridSearchCV columns, ranked by the refit metric."""
    res = pd.DataFrame(gs.cv_results_)
    param_cols = [c for c in res.columns if c.startswith("param_")]
    keep = param_cols + [
        f"mean_train_{refit}",
        f"mean_test_{refit}",
        f"std_test_{refit}",
        f"rank_test_{refit}",
    ]
    keep = [c for c in keep if c in res.columns]
    out = res[keep].sort_values(f"rank_test_{refit}").reset_index(drop=True)
    return out


def evaluate_split(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Predict PD and return classification metrics on a single split."""
    proba = model.predict_proba(X)[:, 1]
    return classification_metrics(y.to_numpy(), proba)


def train_valid_report(
    model: Any,
    data: ModelingData,
    valid_as_holdout: bool = False,
) -> pd.DataFrame:
    """Metrics table on train vs (holdout or internal validation) splits."""
    rows: dict[str, dict[str, float]] = {
        "train": evaluate_split(model, data.X_train, data.y_train)
    }
    if valid_as_holdout:
        rows["holdout"] = evaluate_split(model, data.X_holdout, data.y_holdout)
    return pd.DataFrame(rows).T
