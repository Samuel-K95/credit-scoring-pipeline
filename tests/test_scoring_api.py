"""Tests for the scoring logic and the FastAPI service.

These require a trained champion (`models/champion.joblib`). When it is absent
the tests skip rather than fail, so the suite stays green before training.
"""

from __future__ import annotations

import pytest

from src.modeling.registry import has_model

pytestmark = pytest.mark.skipif(
    not has_model("champion"), reason="champion model not trained yet"
)

EXPECTED_KEYS = {
    "risk_score",
    "score",
    "band",
    "threshold",
    "model",
    "model_version",
    "n_features_supplied",
    "n_features_expected",
    "reason_codes",
}


def test_score_applicant_payload_shape() -> None:
    from src.scoring import score_applicant

    out = score_applicant({"app_ext_source_2": 0.5}, top_reasons=4)
    assert EXPECTED_KEYS <= set(out)
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["band"] in {"approve", "refer", "decline"}
    assert len(out["reason_codes"]) == 4
    for rc in out["reason_codes"]:
        assert rc["direction"] in {"increases_risk", "reduces_risk"}


def test_risk_monotonic_in_external_scores() -> None:
    """Weaker external credit scores should not lower predicted risk."""
    from src.scoring import score_applicant

    strong = {"app_ext_source_1": 0.9, "app_ext_source_2": 0.9, "app_ext_source_3": 0.9}
    weak = {"app_ext_source_1": 0.05, "app_ext_source_2": 0.05, "app_ext_source_3": 0.05}
    assert score_applicant(strong, top_reasons=0)["risk_score"] < (
        score_applicant(weak, top_reasons=0)["risk_score"]
    )


def test_scorecard_points_decrease_with_risk() -> None:
    from src.scoring import _scorecard_points

    assert _scorecard_points(0.1) > _scorecard_points(0.5) > _scorecard_points(0.9)


def test_score_endpoint() -> None:
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        resp = client.post(
            "/score",
            json={"features": {"app_ext_source_2": 0.3}, "top_reasons": 3},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert EXPECTED_KEYS <= set(body)
        assert len(body["reason_codes"]) == 3
