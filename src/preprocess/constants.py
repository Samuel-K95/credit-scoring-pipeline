"""Preprocess constants and artifact paths."""

from __future__ import annotations

from pathlib import Path

from src.config import PROCESSED_DATA_DIR

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
