"""SHAP explainability for the tree champion pipeline.

Tree boosters expose exact SHAP values natively (LightGBM `pred_contrib`,
XGBoost `pred_contribs`, CatBoost `ShapValues`), so global and local
explanations need no extra heavy dependencies. The optional `shap` package is
used only for its beeswarm plot when it happens to be installed; everything
here degrades to matplotlib otherwise.

All SHAP values are in the model's margin (log-odds) space, consistent with
how the boosters score probability of default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.pipeline import Pipeline

from src.modeling.search import MODEL_STEP, PREPROCESS_STEP


@dataclass
class ShapResult:
    """Container for SHAP values aligned to preprocessed features."""

    values: pd.DataFrame  # (n_rows, n_features) margin-space contributions
    features: pd.DataFrame  # the preprocessed feature matrix (same index/cols)
    base_value: float  # expected margin (log-odds) over the background rows

    def global_importance(self) -> pd.Series:
        """Mean absolute SHAP per feature, sorted high to low."""
        imp = self.values.abs().mean(axis=0)
        return imp.sort_values(ascending=False)

    def top_features(self, k: int = 20) -> list[str]:
        return list(self.global_importance().head(k).index)


def split_pipeline(model: Pipeline) -> tuple[Any, Any]:
    """Return (preprocess_step, estimator) from a champion pipeline."""
    if not isinstance(model, Pipeline):
        raise TypeError("Expected an sklearn Pipeline champion.")
    pre = model.named_steps.get(PREPROCESS_STEP)
    est = model.named_steps.get(MODEL_STEP)
    if est is None:
        raise ValueError("Pipeline has no 'model' step.")
    return pre, est


def transform_features(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessing step, returning a named DataFrame."""
    pre, _ = split_pipeline(model)
    if pre is None:
        return X.copy()
    return pre.transform(X)


def _lgbm_contrib(est: Any, Xt: pd.DataFrame) -> np.ndarray:
    return np.asarray(est.booster_.predict(Xt.values, pred_contrib=True))


def _xgb_contrib(est: Any, Xt: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    booster = est.get_booster()
    dmat = xgb.DMatrix(Xt.values, feature_names=list(Xt.columns))
    return np.asarray(booster.predict(dmat, pred_contribs=True))


def _catboost_contrib(est: Any, Xt: pd.DataFrame) -> np.ndarray:
    from catboost import Pool

    pool = Pool(Xt.values)
    return np.asarray(est.get_feature_importance(data=pool, type="ShapValues"))


def _native_contrib(est: Any, Xt: pd.DataFrame) -> np.ndarray:
    """Return an (n_rows, n_features + 1) contribution matrix (last col = base)."""
    name = type(est).__name__
    if name == "LGBMClassifier":
        return _lgbm_contrib(est, Xt)
    if name == "XGBClassifier":
        return _xgb_contrib(est, Xt)
    if name == "CatBoostClassifier":
        return _catboost_contrib(est, Xt)
    raise NotImplementedError(
        f"No native SHAP path for {name}; install shap for a model-agnostic"
        " explainer."
    )


def compute_shap(model: Pipeline, X: pd.DataFrame) -> ShapResult:
    """Compute exact tree SHAP values for the given rows.

    The returned values are in margin (log-odds) space and reconstruct the raw
    score as ``values.sum(axis=1) + base_value``.
    """
    _, est = split_pipeline(model)
    Xt = transform_features(model, X)
    contrib = _native_contrib(est, Xt)
    shap_vals = contrib[:, :-1]
    base = float(np.mean(contrib[:, -1]))
    values = pd.DataFrame(shap_vals, columns=Xt.columns, index=Xt.index)
    return ShapResult(values=values, features=Xt, base_value=base)


# ---------------------------------------------------------------------------
# Plotting (matplotlib only, no shap dependency required)
# ---------------------------------------------------------------------------


def plot_global_bar(result: ShapResult, ax: Axes, top: int = 20) -> Axes:
    """Horizontal bar of mean absolute SHAP for the top features."""
    imp = result.global_importance().head(top).iloc[::-1]
    ax.barh(imp.index, imp.values, color="#2b6cb0")
    ax.set_xlabel("mean(|SHAP|)  (log-odds)")
    ax.set_title(f"Global feature importance (top {top})")
    ax.margins(y=0.01)
    return ax


def plot_beeswarm(result: ShapResult, ax: Axes, top: int = 15) -> Axes:
    """Compact beeswarm: SHAP spread per feature, colored by feature value."""
    feats = result.top_features(top)[::-1]
    rng = np.random.default_rng(0)
    for row, feat in enumerate(feats):
        sv = result.values[feat].to_numpy()
        fv = result.features[feat].to_numpy().astype(float)
        lo, hi = np.nanpercentile(fv, [1, 99])
        norm = np.clip((fv - lo) / (hi - lo + 1e-12), 0, 1)
        jitter = rng.uniform(-0.18, 0.18, size=sv.shape[0])
        ax.scatter(
            sv,
            np.full_like(sv, row) + jitter,
            c=norm,
            cmap="coolwarm",
            s=6,
            alpha=0.5,
            linewidths=0,
        )
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.axvline(0.0, color="#444444", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds; red = higher feature value)")
    ax.set_title(f"SHAP summary (top {top})")
    return ax


def plot_dependence(result: ShapResult, feature: str, ax: Axes) -> Axes:
    """Scatter of a feature value vs its SHAP contribution."""
    fv = result.features[feature].to_numpy().astype(float)
    sv = result.values[feature].to_numpy()
    ax.scatter(fv, sv, s=8, alpha=0.4, color="#2b6cb0", linewidths=0)
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value (log-odds)")
    ax.set_title(f"Dependence: {feature}")
    return ax


def plot_waterfall(
    result: ShapResult, row: int, ax: Axes, top: int = 12
) -> Axes:
    """Local explanation: largest signed contributions for one applicant."""
    sv = result.values.iloc[row]
    order = sv.abs().sort_values(ascending=False).head(top).index
    contrib = sv[order].iloc[::-1]
    feat_vals = result.features.iloc[row][order].iloc[::-1]
    labels = [f"{f} = {v:.3g}" for f, v in zip(order[::-1], feat_vals.values)]
    colors = ["#c53030" if c > 0 else "#2f855a" for c in contrib.values]
    ax.barh(range(len(contrib)), contrib.values, color=colors)
    ax.set_yticks(range(len(contrib)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#444444", lw=0.8)
    margin = float(result.base_value + sv.sum())
    proba = 1.0 / (1.0 + np.exp(-margin))
    ax.set_xlabel("SHAP value (log-odds; red raises default risk)")
    ax.set_title(f"Local explanation (PD = {proba:.3f})")
    return ax
