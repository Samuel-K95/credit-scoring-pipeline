"""Shared scoring logic used by the API and tests.

The champion is an sklearn pipeline (preprocess + booster), so scoring only
needs a feature mapping aligned to the training columns; the pipeline imputes,
caps, and adds missing flags internally.

Note on the probability: the champion trains with class weighting
(`scale_pos_weight`), so its raw output is a well-ranked but uncalibrated risk
score in [0, 1], not a calibrated probability of default. It is reported as
`risk_score`, turned into a monotonic scorecard `score`, and mapped to a
decision `band` relative to the tuned threshold. Reason codes come from exact
tree SHAP values for the single applicant.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from src.config import (
    BAND_APPROVE_FRACTION,
    MODEL_VERSION,
    SCORECARD_BASE_SCORE,
    SCORECARD_PDO,
)
from src.modeling import compute_shap, load_model
from src.modeling.registry import REGISTRY_PATH

DEFAULT_THRESHOLD = 0.5


def _scorecard_points(risk: float) -> int:
    """Map a risk score in (0, 1) to scorecard points (higher = safer).

    Uses standard points-to-double-the-odds scaling anchored so that a model
    output of 0.5 (odds 1:1) yields the base score.
    """
    eps = 1e-6
    risk = min(max(risk, eps), 1.0 - eps)
    odds_bad = risk / (1.0 - risk)
    factor = SCORECARD_PDO / math.log(2.0)
    points = SCORECARD_BASE_SCORE - factor * math.log(odds_bad)
    return int(round(points))


@dataclass
class ReasonCode:
    feature: str
    value: float
    contribution: float  # signed SHAP (log-odds); positive raises risk
    direction: str  # "increases_risk" or "reduces_risk"

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round(float(self.value), 6),
            "contribution": round(float(self.contribution), 6),
            "direction": self.direction,
        }


@dataclass
class Scorer:
    """Loaded champion plus the metadata needed to score one applicant."""

    model: Any
    feature_cols: list[str]
    threshold: float
    model_version: str
    source_model: str

    def _frame(self, features: dict[str, Any]) -> pd.DataFrame:
        """Build a 1-row frame aligned to training columns (missing -> NaN)."""
        row = {col: features.get(col, float("nan")) for col in self.feature_cols}
        return pd.DataFrame([row], columns=self.feature_cols)

    def band_for(self, risk: float) -> str:
        if risk >= self.threshold:
            return "decline"
        if risk >= self.threshold * BAND_APPROVE_FRACTION:
            return "refer"
        return "approve"

    def reason_codes(self, X: pd.DataFrame, top: int = 5) -> list[ReasonCode]:
        result = compute_shap(self.model, X)
        sv = result.values.iloc[0]
        feats = sv.abs().sort_values(ascending=False).head(top).index
        codes: list[ReasonCode] = []
        for feat in feats:
            contrib = float(sv[feat])
            codes.append(
                ReasonCode(
                    feature=feat,
                    value=float(result.features.iloc[0][feat]),
                    contribution=contrib,
                    direction="increases_risk" if contrib > 0 else "reduces_risk",
                )
            )
        return codes

    def score(self, features: dict[str, Any], top_reasons: int = 5) -> dict[str, Any]:
        """Return risk score, scorecard points, decision band, and reasons."""
        if not isinstance(features, dict):
            raise TypeError("features must be a mapping of feature name to value.")
        X = self._frame(features)
        risk = float(self.model.predict_proba(X)[:, 1][0])
        reasons = self.reason_codes(X, top=top_reasons) if top_reasons > 0 else []
        return {
            "risk_score": round(risk, 6),
            "score": _scorecard_points(risk),
            "band": self.band_for(risk),
            "threshold": round(self.threshold, 4),
            "model": self.source_model,
            "model_version": self.model_version,
            "n_features_supplied": sum(1 for c in self.feature_cols if c in features),
            "n_features_expected": len(self.feature_cols),
            "reason_codes": [r.as_dict() for r in reasons],
        }


def _champion_meta() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(encoding="ascii"))
        return reg.get("champion", {})
    return {}


@lru_cache(maxsize=1)
def get_scorer() -> Scorer:
    """Load and cache the champion scorer (model file + registry metadata)."""
    try:
        model = load_model("champion")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No champion model found. Train it first: python -m src.train"
        ) from exc
    meta = _champion_meta()
    feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols and hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    threshold = float(meta.get("metrics", {}).get("best_f1_threshold", DEFAULT_THRESHOLD))
    return Scorer(
        model=model,
        feature_cols=feature_cols,
        threshold=threshold,
        model_version=str(meta.get("model_version", MODEL_VERSION)),
        source_model=str(meta.get("source_model", "champion")),
    )


def score_applicant(features: dict[str, Any], top_reasons: int = 5) -> dict[str, Any]:
    """Score one applicant feature mapping and return the decision payload."""
    return get_scorer().score(features, top_reasons=top_reasons)
