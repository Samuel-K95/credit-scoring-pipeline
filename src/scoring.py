"""Shared scoring logic used by the API and tests."""

from __future__ import annotations

from typing import Any


def score_applicant(features: dict[str, Any]) -> dict[str, Any]:
    """Return PD, score, decision band, and reason codes.

    Args:
        features: Applicant feature mapping available at application time.

    Returns:
        Scoring payload. Implementation follows model training.
    """
    raise NotImplementedError("Scoring function not implemented yet.")
