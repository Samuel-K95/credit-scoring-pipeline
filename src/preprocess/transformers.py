"""Train-fit-only sklearn transformers for numeric feature tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocess.constants import HIGH_SIGNAL_MISSING


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Add binary missing flags for selected columns; keep original NaNs."""

    def __init__(self, columns: list[str] | tuple[str, ...] | None = None):
        # Store the argument unchanged so the estimator is clone-safe
        # (sklearn requires __init__ not to modify its parameters).
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> MissingIndicatorAdder:
        cols = list(self.columns) if self.columns is not None else list(HIGH_SIGNAL_MISSING)
        self.columns_ = [c for c in cols if c in X.columns]
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
