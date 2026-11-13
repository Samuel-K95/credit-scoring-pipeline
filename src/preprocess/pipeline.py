"""Fit preprocess on train, persist splits and the fitted pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    SPLIT_INDICES_PATH,
    TARGET_COL,
    TRAIN_FEATURES_PATH,
)
from src.preprocess.constants import HIGH_SIGNAL_MISSING, PIPELINE_PATH
from src.preprocess.splits import feature_columns, make_splits
from src.preprocess.transformers import build_preprocess_pipeline


def run_preprocess(
    features_path: Path = TRAIN_FEATURES_PATH,
    splits_path: Path = SPLIT_INDICES_PATH,
    pipeline_path: Path = PIPELINE_PATH,
) -> dict[str, Any]:
    """Build splits, fit preprocess on train, and persist artifacts."""
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run: python -m src.features"
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
