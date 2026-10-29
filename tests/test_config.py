"""Smoke tests for project configuration."""

from __future__ import annotations

from src.config import (
    FEATURE_PREFIXES,
    HOLDOUT_SIZE,
    ID_COL,
    N_CV_FOLDS,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COL,
)


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src" / "config.py").is_file()


def test_core_constants() -> None:
    assert ID_COL == "SK_ID_CURR"
    assert TARGET_COL == "TARGET"
    assert RANDOM_SEED == 42
    assert 0.0 < HOLDOUT_SIZE < 1.0
    assert N_CV_FOLDS >= 2
    assert FEATURE_PREFIXES == ("inst_", "bur_", "prev_", "pos_", "cc_", "app_")
