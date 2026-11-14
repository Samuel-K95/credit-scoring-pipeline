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


def test_modeling_package_exports() -> None:
    from src.modeling import (
        ModelingData,
        classification_metrics,
        cv_results_frame,
        load_modeling_data,
        make_pipeline,
        run_grid_search,
        subsample,
    )

    assert ModelingData.__name__ == "ModelingData"
    for fn in (
        classification_metrics,
        cv_results_frame,
        load_modeling_data,
        make_pipeline,
        run_grid_search,
        subsample,
    ):
        assert callable(fn)


def test_modeling_submodules_import() -> None:
    from src.modeling import boosting, curves, linear, plots

    assert callable(plots.plot_roc_pr)
    assert callable(plots.plot_loss_curve)
    assert callable(curves.param_loss_curve)
    assert callable(linear.fit_logreg_sgd_curve)
    assert callable(boosting.fit_xgb_curve)


def test_champions_and_registry() -> None:
    from src.modeling import (
        build_all_champions,
        champion_names,
        has_model,
        list_models,
        load_model,
        save_model,
    )

    names = champion_names()
    assert {"logistic_regression", "random_forest", "xgboost", "catboost"} <= set(names)
    champs = build_all_champions(pos_weight=10.0)
    assert set(champs) == set(names)
    for est in champs.values():
        assert hasattr(est, "fit")
    for fn in (has_model, list_models, load_model, save_model):
        assert callable(fn)


def test_explain_exports_and_shap_reconstruction() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        import pytest

        pytest.skip("lightgbm not installed")

    from src.modeling import compute_shap, plot_global_bar, transform_features

    for fn in (compute_shap, plot_global_bar, transform_features):
        assert callable(fn)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = (X["a"] + X["b"] > 0).astype(int)
    pipe = Pipeline(
        [
            ("preprocess", FunctionTransformer()),
            ("model", LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1)),
        ]
    )
    pipe.fit(X, y)

    result = compute_shap(pipe, X)
    assert result.values.shape == X.shape
    margin = result.values.sum(axis=1).to_numpy() + result.base_value
    recon = 1.0 / (1.0 + np.exp(-margin))
    proba = pipe.predict_proba(X)[:, 1]
    assert np.max(np.abs(recon - proba)) < 1e-6
    assert len(result.global_importance()) == X.shape[1]


def test_classification_metrics_values() -> None:
    import numpy as np

    from src.modeling import classification_metrics

    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    m = classification_metrics(y, s)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["gini"] == 2 * m["roc_auc"] - 1
    assert 0.0 <= m["ks"] <= 1.0
