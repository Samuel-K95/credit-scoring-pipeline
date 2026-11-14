"""Cutoff economics and segment discrimination on synthetic scores."""

from __future__ import annotations

import numpy as np

from src.modeling.metrics import cutoff_table, grouped_discrimination


def test_cutoff_approval_rises_with_threshold() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    rows = cutoff_table(y, score, thresholds=(0.3, 0.5, 0.7), lgd=1.0)
    rates = [row["approval_rate"] for row in rows]
    assert rates == sorted(rates)
    # score < 0.3 approves the two lowest scores, both goods.
    assert rows[0]["n_approved"] == 2
    assert rows[0]["bad_rate_approved"] == 0.0
    assert rows[0]["expected_loss_proxy"] == 0.0


def test_cutoff_loss_proxy_scales_with_lgd() -> None:
    y = np.array([1, 0, 1, 0])
    score = np.array([0.2, 0.2, 0.2, 0.2])
    base = cutoff_table(y, score, thresholds=(0.5,), lgd=1.0)[0]
    half = cutoff_table(y, score, thresholds=(0.5,), lgd=0.5)[0]
    assert base["approval_rate"] == 1.0
    assert base["bad_rate_approved"] == 0.5
    assert half["expected_loss_proxy"] == base["expected_loss_proxy"] * 0.5


def test_grouped_discrimination_skips_single_class() -> None:
    y = np.array([0, 1, 0, 1, 0, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    groups = np.array(["a", "a", "a", "a", "b", "b"])
    rows = {row["segment"]: row for row in grouped_discrimination(y, score, groups)}
    assert rows["a"]["roc_auc"] == 1.0
    assert rows["b"]["roc_auc"] is None
    assert rows["b"]["n"] == 2
    assert rows["b"]["default_rate"] == 0.0
