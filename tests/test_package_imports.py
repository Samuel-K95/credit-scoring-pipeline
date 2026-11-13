"""Smoke tests that Stage 1-4 packages import cleanly."""

from __future__ import annotations


def test_data_quality_package_exports() -> None:
    from src.data_quality import (
        Decision,
        clean_application,
        clean_bureau,
        clean_previous_application,
        run_data_quality,
    )

    assert Decision.__name__ == "Decision"
    assert callable(clean_application)
    assert callable(clean_bureau)
    assert callable(clean_previous_application)
    assert callable(run_data_quality)


def test_features_package_exports() -> None:
    from src.features import SQL_FILES, build_features

    assert len(SQL_FILES) == 5
    assert callable(build_features)


def test_preprocess_package_exports() -> None:
    from src.preprocess import (
        HIGH_SIGNAL_MISSING,
        build_preprocess_pipeline,
        make_splits,
        run_preprocess,
    )

    assert len(HIGH_SIGNAL_MISSING) >= 1
    assert callable(build_preprocess_pipeline)
    assert callable(make_splits)
    assert callable(run_preprocess)


def test_io_utils_exports() -> None:
    from src.io_utils import df_to_markdown, replace_sentinel, write_json

    assert callable(write_json)
    assert callable(df_to_markdown)
    assert callable(replace_sentinel)
