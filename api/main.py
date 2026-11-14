"""FastAPI entrypoint for health and scoring endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_VERSION
from src.scoring import get_scorer, score_applicant


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the champion once at startup so the first request is fast."""
    try:
        get_scorer()
    except RuntimeError:
        # Allow the service to boot (health stays up) even without a model;
        # /score will return a clear 503 until the champion is trained.
        pass
    yield


app = FastAPI(title="Credit Scoring API", version=MODEL_VERSION, lifespan=lifespan)


class ScoreRequest(BaseModel):
    """Applicant features available at application time.

    `features` maps engineered feature names (see the model card) to numeric
    values. Unknown or omitted features are imputed by the model pipeline.
    """

    features: dict[str, float] = Field(default_factory=dict)
    top_reasons: int = Field(default=5, ge=0, le=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "app_ext_source_1": 0.5,
                    "app_ext_source_2": 0.42,
                    "app_ext_source_3": 0.31,
                    "app_credit_to_income": 3.1,
                    "app_amt_annuity": 24700.0,
                },
                "top_reasons": 5,
            }
        }
    }


class ReasonCodeOut(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    risk_score: float
    score: int
    band: str
    threshold: float
    model: str
    model_version: str
    n_features_supplied: int
    n_features_expected: int
    reason_codes: list[ReasonCodeOut]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> dict[str, Any]:
    """Score an applicant and return risk, scorecard points, and reason codes."""
    try:
        return score_applicant(request.features, top_reasons=request.top_reasons)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
