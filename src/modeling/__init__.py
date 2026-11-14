"""Stage 5 modelling toolkit: reusable data, metrics, plots, and search helpers."""

from src.modeling.champions import (
    SEARCH_CV_AUC,
    build_all_champions,
    build_champion,
    champion_names,
)
from src.modeling.data import ModelingData, load_modeling_data, subsample
from src.modeling.explain import (
    ShapResult,
    compute_shap,
    plot_beeswarm,
    plot_dependence,
    plot_global_bar,
    plot_waterfall,
    transform_features,
)
from src.modeling.metrics import (
    classification_metrics,
    cutoff_table,
    grouped_discrimination,
    ks_statistic,
    metrics_frame,
)
from src.modeling.registry import (
    has_model,
    list_models,
    load_all_models,
    load_model,
    registry_metrics,
    save_model,
)
from src.modeling.search import (
    cv_results_frame,
    evaluate_split,
    make_pipeline,
    prefix_grid,
    run_grid_search,
    train_valid_report,
)

__all__ = [
    "ModelingData",
    "ShapResult",
    "SEARCH_CV_AUC",
    "compute_shap",
    "plot_beeswarm",
    "plot_dependence",
    "plot_global_bar",
    "plot_waterfall",
    "transform_features",
    "build_all_champions",
    "build_champion",
    "champion_names",
    "classification_metrics",
    "cutoff_table",
    "grouped_discrimination",
    "cv_results_frame",
    "evaluate_split",
    "has_model",
    "ks_statistic",
    "list_models",
    "load_all_models",
    "load_model",
    "load_modeling_data",
    "make_pipeline",
    "metrics_frame",
    "prefix_grid",
    "registry_metrics",
    "run_grid_search",
    "save_model",
    "subsample",
    "train_valid_report",
]
