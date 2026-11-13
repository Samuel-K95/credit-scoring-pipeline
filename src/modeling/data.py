"""Load modelling matrices and reproducible CV folds for Stage 5 notebooks.

The holdout partition is never touched during model selection; it is returned
only so the final chosen model can be evaluated once. Cross-validation uses the
train partition with the same stratified folds saved by `src.preprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    N_CV_FOLDS,
    RANDOM_SEED,
    TARGET_COL,
    TRAIN_FEATURES_PATH,
)
from src.preprocess import load_splits


@dataclass
class ModelingData:
    """Container for train / holdout matrices and CV folds.

    X_train / y_train are the model-selection partition (fit + CV here).
    X_holdout / y_holdout are reserved for a single final evaluation.
    cv is a list of (train_pos, valid_pos) integer-position arrays into
    X_train for use with scikit-learn `cv=`.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    feature_cols: list[str]
    cv: list[tuple[np.ndarray, np.ndarray]]

    @property
    def pos_weight(self) -> float:
        """Negative/positive ratio for cost-sensitive learners."""
        pos = float(self.y_train.sum())
        neg = float(len(self.y_train) - pos)
        return neg / max(pos, 1.0)

    def summary(self) -> dict[str, object]:
        return {
            "n_train": int(len(self.X_train)),
            "n_holdout": int(len(self.X_holdout)),
            "n_features": len(self.feature_cols),
            "train_target_rate": round(float(self.y_train.mean()), 4),
            "holdout_target_rate": round(float(self.y_holdout.mean()), 4),
            "n_folds": len(self.cv),
            "pos_weight": round(self.pos_weight, 2),
        }


def load_modeling_data(features_path=TRAIN_FEATURES_PATH) -> ModelingData:
    """Load features, apply the saved split, and build positional CV folds.

    Preprocessing (impute/cap/flags) is intentionally NOT applied here. Wrap the
    estimator in a pipeline with `src.preprocess.build_preprocess_pipeline()` so
    that preprocessing is refit inside every CV fold (no leakage).
    """
    splits = load_splits()
    df = pd.read_parquet(features_path)
    feature_cols = list(splits["feature_cols"])

    train_idx = np.asarray(splits["train_idx"], dtype=int)
    holdout_idx = np.asarray(splits["holdout_idx"], dtype=int)

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_holdout = X.iloc[holdout_idx].reset_index(drop=True)
    y_holdout = y.iloc[holdout_idx].reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=splits.get("n_cv_folds", N_CV_FOLDS),
        shuffle=True,
        random_state=splits.get("random_seed", RANDOM_SEED),
    )
    cv = [(tr, va) for tr, va in skf.split(X_train, y_train)]

    return ModelingData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        feature_cols=feature_cols,
        cv=cv,
    )


def subsample(
    data: ModelingData,
    frac: float,
    seed: int = RANDOM_SEED,
) -> ModelingData:
    """Return a stratified sub-sample of the train partition for fast search.

    Useful for keeping GridSearchCV tractable while iterating. The holdout and
    feature columns are preserved; CV folds are rebuilt on the sampled train.
    """
    if frac >= 1.0:
        return data
    rng = np.random.default_rng(seed)
    y = data.y_train
    keep: list[int] = []
    for cls in (0, 1):
        pos = np.where(y.to_numpy() == cls)[0]
        n = max(1, int(round(len(pos) * frac)))
        keep.extend(rng.choice(pos, size=n, replace=False).tolist())
    keep_arr = np.sort(np.asarray(keep, dtype=int))

    X_train = data.X_train.iloc[keep_arr].reset_index(drop=True)
    y_train = data.y_train.iloc[keep_arr].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=len(data.cv), shuffle=True, random_state=seed)
    cv = [(tr, va) for tr, va in skf.split(X_train, y_train)]
    return ModelingData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=data.X_holdout,
        y_holdout=data.y_holdout,
        feature_cols=data.feature_cols,
        cv=cv,
    )
