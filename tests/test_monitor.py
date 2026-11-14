"""Tests for PSI / CSI drift monitoring math (no model or data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitor import (
    drift_status,
    feature_psi,
    population_stability_index,
)


def test_psi_zero_for_identical_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_grows_with_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5000)
    small = population_stability_index(ref, ref + 0.2)
    large = population_stability_index(ref, ref + 1.5)
    assert 0.0 < small < large


def test_psi_handles_nan_and_low_cardinality() -> None:
    ref = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    cur = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])
    assert population_stability_index(ref, cur) >= 0.0
    # single unique value cannot form bins -> 0.0
    assert population_stability_index(np.ones(100), np.ones(100)) == 0.0


def test_drift_status_bands() -> None:
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "watch"
    assert drift_status(0.40) == "alert"


def test_feature_psi_table_sorted_desc() -> None:
    rng = np.random.default_rng(2)
    ref = pd.DataFrame(
        {"a": rng.normal(size=2000), "b": rng.normal(size=2000)}
    )
    cur = pd.DataFrame(
        {"a": rng.normal(size=2000), "b": rng.normal(size=2000) + 2.0}
    )
    table = feature_psi(ref, cur, ["a", "b"])
    assert list(table.columns) == ["feature", "psi", "status"]
    assert table.iloc[0]["feature"] == "b"  # larger shift ranks first
    assert (table["psi"].values[:-1] >= table["psi"].values[1:]).all()
