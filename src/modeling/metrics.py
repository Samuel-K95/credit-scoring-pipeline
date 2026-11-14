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


# Illustrative loss-given-default used by the cutoff table. Exposure is 1
# currency unit per approved loan. This is not a priced expected-loss model.
DEFAULT_LGD: float = 1.0

DEFAULT_CUTOFFS: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65, 0.80)


def cutoff_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | tuple[float, ...] | None = None,
    lgd: float = DEFAULT_LGD,
) -> list[dict[str, float | int]]:
    """Approval rate, bad rate, and an illustrative loss proxy by cutoff.

    `y_score` is a risk score where higher means more likely to default.
    An applicant is approved when `y_score < threshold`.

    Assumptions (illustrative only):
    - each approved loan has exposure of 1
    - loss given default is the constant `lgd` (default 1)
    - no interest income, recoveries beyond LGD, or prepayment
    The proxy is `approval_rate * bad_rate_approved * lgd`, the share of the
    book that defaults among approved loans, scaled by LGD.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if thresholds is None:
        thresholds = DEFAULT_CUTOFFS
    n = len(y_true)
    rows: list[dict[str, float]] = []
    for raw_t in thresholds:
        t = float(raw_t)
        approved = y_score < t
        n_approved = int(approved.sum())
        approval_rate = (n_approved / n) if n else 0.0
        if n_approved:
            bad_rate = float(y_true[approved].mean())
            loss = approval_rate * bad_rate * float(lgd)
        else:
            bad_rate = 0.0
            loss = 0.0
        rows.append(
            {
                "threshold": t,
                "approval_rate": approval_rate,
                "n_approved": n_approved,
                "bad_rate_approved": bad_rate,
                "expected_loss_proxy": loss,
            }
        )
    return rows


def grouped_discrimination(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
) -> list[dict[str, float | int | str | None]]:
    """ROC AUC and default rate inside each segment label.

    Segments with a single class return a null ROC AUC rather than a fake one.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)
    rows: list[dict[str, float | int | str | None]] = []
    # Preserve first-seen order.
    seen: list[str] = []
    for raw in groups:
        label = str(raw)
        if label not in seen and label != "nan":
            seen.append(label)
    for label in seen:
        mask = np.array([str(g) == label for g in groups])
        yt = y_true[mask]
        ys = y_score[mask]
        n = int(mask.sum())
        n_pos = int(yt.sum()) if n else 0
        auc: float | None
        if n_pos == 0 or n_pos == n or n < 2:
            auc = None
        else:
            auc = float(roc_auc_score(yt, ys))
        rows.append(
            {
                "segment": label,
                "n": n,
                "n_pos": n_pos,
                "default_rate": float(yt.mean()) if n else 0.0,
                "roc_auc": auc,
            }
        )
    return rows
