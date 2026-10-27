"""FastAPI entrypoint for health and scoring endpoints (stub)."""

from __future__ import annotations

from fastapi import FastAPI

from src.config import MODEL_VERSION

app = FastAPI(title="Credit Scoring API", version=MODEL_VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/score")
def score() -> dict[str, str]:
    """Score an applicant. Implemented in a later stage."""
    return {
        "status": "not_implemented",
        "model_version": MODEL_VERSION,
        "detail": "Scoring logic will be wired after training.",
    }
