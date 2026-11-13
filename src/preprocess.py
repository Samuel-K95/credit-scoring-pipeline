"""Preprocessing pipeline and stratified train/holdout / CV splits.

Ported from notebooks/05_preprocess_and_splits.ipynb.

All transformers are fit on training rows only. Categorical encoding is not
needed here because Stage-3 SQL already aggregates categoricals into numeric
rates and counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    HOLDOUT_SIZE,
    ID_COL,
    N_CV_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SPLIT_INDICES_PATH,
    TARGET_COL,
    TRAIN_FEATURES_PATH,
)

# High-signal / high-missing fields get explicit missing indicators.
HIGH_SIGNAL_MISSING: tuple[str, ...] = (
    "app_ext_source_1",
    "app_ext_source_2",
    "app_ext_source_3",
    "app_years_employed",
    "cc_utilisation_mean",
    "cc_utilisation_max",
    "inst_late_rate",
    "inst_late_rate_3m",
    "inst_late_rate_6m",
    "inst_late_rate_12m",
    "bur_utilisation",
    "bur_bb_dpd_share_mean",
    "prev_refused_rate",
    "prev_approved_rate",
)

PIPELINE_PATH: Path = PROCESSED_DATA_DIR / "preprocess_pipeline.joblib"


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Add binary missing flags for selected columns; keep original NaNs."""

    def __init__(self, columns: list[str] | tuple[str, ...] | None = None):
        self.columns = list(columns) if columns is not None else list(HIGH_SIGNAL_MISSING)

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingIndicatorAdder:
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in self.columns_:
            out[f"{c}_was_missing"] = out[c].isna().astype("int8")
        return out


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns using training quantiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y: Any = None) -> QuantileCapper:
        num = X.select_dtypes(include=[np.number])
        cols = [c for c in num.columns if not str(c).endswith("_was_missing")]
        self.bounds_: dict[str, tuple[float, float]] = {}
        for c in cols:
            lo, hi = num[c].quantile([self.lower, self.upper])
            self.bounds_[c] = (float(lo), float(hi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in out.columns:
                out[c] = out[c].clip(lo, hi)
        return out


class MedianImputerFrame(BaseEstimator, TransformerMixin):
    """Median-impute all columns; return a DataFrame with the same columns."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> MedianImputerFrame:
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = self.imputer_.transform(X[self.columns_])
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)


def build_preprocess_pipeline(
    missing_cols: list[str] | tuple[str, ...] | None = None,
) -> Pipeline:
    """Return the sklearn preprocess pipeline (unfitted)."""
    cols = list(missing_cols) if missing_cols is not None else list(HIGH_SIGNAL_MISSING)
    return Pipeline(
        steps=[
            ("missing_flags", MissingIndicatorAdder(cols)),
            ("impute", MedianImputerFrame()),
            ("cap", QuantileCapper(lower=0.01, upper=0.99)),
        ]
    )


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


def run_preprocess(
    features_path: Path = TRAIN_FEATURES_PATH,
    splits_path: Path = SPLIT_INDICES_PATH,
    pipeline_path: Path = PIPELINE_PATH,
) -> dict[str, Any]:
    """Build splits, fit preprocess on train, and persist artifacts."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run: python -m src.build_features"
        )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(features_path)
    feat_cols = feature_columns(df)
    X = df[feat_cols]
    y = df[TARGET_COL].astype(int)

    splits = make_splits(y)
    splits["feature_cols"] = feat_cols
    missing_cols = [c for c in HIGH_SIGNAL_MISSING if c in feat_cols]
    splits["high_signal_missing"] = missing_cols

    train_idx = np.asarray(splits["train_idx"])
    holdout_idx = np.asarray(splits["holdout_idx"])

    pipe = build_preprocess_pipeline(missing_cols)
    pipe.fit(X.iloc[train_idx])
    X_train_t = pipe.transform(X.iloc[train_idx])
    X_hold_t = pipe.transform(X.iloc[holdout_idx])

    if int(X_train_t.isna().sum().sum()) != 0 or int(X_hold_t.isna().sum().sum()) != 0:
        raise ValueError("Preprocess left NaNs after transform")

    splits_path.write_text(json.dumps(splits), encoding="ascii")
    joblib.dump(pipe, pipeline_path)

    summary = {
        "n_rows": int(len(df)),
        "n_features_in": len(feat_cols),
        "n_features_out": int(X_train_t.shape[1]),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
        "train_target_rate": splits["train_target_rate"],
        "holdout_target_rate": splits["holdout_target_rate"],
        "splits_path": str(splits_path),
        "pipeline_path": str(pipeline_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_splits(path: Path = SPLIT_INDICES_PATH) -> dict[str, Any]:
    """Load saved split indices."""
    return json.loads(path.read_text(encoding="ascii"))


def load_pipeline(path: Path = PIPELINE_PATH) -> Pipeline:
    """Load the fitted preprocess pipeline."""
    return joblib.load(path)


def main() -> None:
    """CLI entrypoint."""
    run_preprocess()


if __name__ == "__main__":
    # Re-enter via package import so joblib pickles classes as src.preprocess.*
    # (not __main__.*) when this file is run with `python -m src.preprocess`.
    from src.preprocess import run_preprocess as _run

    _run()
