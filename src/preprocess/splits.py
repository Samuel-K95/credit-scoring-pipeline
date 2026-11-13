"""Stratified holdout and CV index helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import HOLDOUT_SIZE, ID_COL, N_CV_FOLDS, RANDOM_SEED, TARGET_COL


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric modelling columns (exclude id and target)."""
    return [c for c in df.columns if c not in {ID_COL, TARGET_COL}]


def make_splits(
    y: pd.Series,
    *,
    holdout_size: float = HOLDOUT_SIZE,
    n_folds: int = N_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Create stratified holdout indices and CV folds on the train partition."""
    idx = np.arange(len(y))
    train_idx, holdout_idx = train_test_split(
        idx,
        test_size=holdout_size,
        stratify=y,
        random_state=seed,
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_folds: list[dict[str, Any]] = []
    for fold_i, (tr, va) in enumerate(skf.split(train_idx, y.iloc[train_idx])):
        cv_folds.append(
            {
                "fold": fold_i,
                "train_idx": train_idx[tr].tolist(),
                "valid_idx": train_idx[va].tolist(),
            }
        )
    return {
        "random_seed": seed,
        "holdout_size": holdout_size,
        "n_cv_folds": n_folds,
        "n_rows": int(len(y)),
        "train_idx": train_idx.tolist(),
        "holdout_idx": holdout_idx.tolist(),
        "cv_folds": cv_folds,
        "train_target_rate": float(y.iloc[train_idx].mean()),
        "holdout_target_rate": float(y.iloc[holdout_idx].mean()),
    }
