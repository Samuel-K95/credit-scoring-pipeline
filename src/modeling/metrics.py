"""Discrimination and calibration metrics for PD models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between good/bad score distributions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_sorted = y_true[order]
    pos_total = y_sorted.sum()
    neg_total = len(y_sorted) - pos_total
    if pos_total == 0 or neg_total == 0:
        return 0.0
    cum_pos = np.cumsum(y_sorted) / pos_total
    cum_neg = np.cumsum(1 - y_sorted) / neg_total
    return float(np.max(np.abs(cum_pos - cum_neg)))


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximising F1 over a coarse grid."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Core PD metrics: AUC, Gini, KS, PR-AUC, log-loss, Brier, best-F1."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = roc_auc_score(y_true, y_score)
    best_t, best_f1 = best_threshold_by_f1(y_true, y_score)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_score)),
        "best_f1": best_f1,
        "best_f1_threshold": best_t,
    }


def metrics_frame(rows: dict[str, dict[str, float]]):
    """Build a tidy metrics DataFrame from {label: metrics} mappings."""
    import pandas as pd

    return pd.DataFrame(rows).T
