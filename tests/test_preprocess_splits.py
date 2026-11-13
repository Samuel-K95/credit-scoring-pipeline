"""Smoke tests for stratified splits and preprocess pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COL, RANDOM_SEED, TARGET_COL, TRAIN_FEATURES_PATH
from src.preprocess import (
    PIPELINE_PATH,
    build_preprocess_pipeline,
    feature_columns,
    make_splits,
)


def test_make_splits_stratified_and_disjoint() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    splits = make_splits(y, holdout_size=0.2, n_folds=5, seed=RANDOM_SEED)
    train = set(splits["train_idx"])
    hold = set(splits["holdout_idx"])
    assert train.isdisjoint(hold)
    assert train | hold == set(range(len(y)))
    assert abs(splits["train_target_rate"] - splits["holdout_target_rate"]) < 0.05
    assert len(splits["cv_folds"]) == 5
    for fold in splits["cv_folds"]:
        assert set(fold["train_idx"]).isdisjoint(set(fold["valid_idx"]))
        assert set(fold["train_idx"]) | set(fold["valid_idx"]) == train


def test_preprocess_fit_transform_no_nans() -> None:
    if not TRAIN_FEATURES_PATH.exists():
        return  # skip if features not built in this environment
    df = pd.read_parquet(TRAIN_FEATURES_PATH)
    # Tiny stratified sample for speed
    parts = [
        g.sample(n=min(200, len(g)), random_state=RANDOM_SEED)
        for _, g in df.groupby(TARGET_COL, sort=False)
    ]
    sample = pd.concat(parts, ignore_index=True)
    cols = feature_columns(sample)
    X = sample[cols]
    y = sample[TARGET_COL].astype(int)
    splits = make_splits(y, holdout_size=0.25, n_folds=3, seed=RANDOM_SEED)
    pipe = build_preprocess_pipeline()
    pipe.fit(X.iloc[splits["train_idx"]])
    Xt = pipe.transform(X.iloc[splits["train_idx"]])
    Xh = pipe.transform(X.iloc[splits["holdout_idx"]])
    assert int(Xt.isna().sum().sum()) == 0
    assert int(Xh.isna().sum().sum()) == 0
    assert Xt.shape[1] >= X.shape[1]
