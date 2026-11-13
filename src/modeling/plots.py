"""Reusable train/validation plots for the Stage 5 modelling notebooks."""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import learning_curve, validation_curve

from src.modeling.metrics import classification_metrics


def plot_roc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str = "model",
    axes: Sequence[plt.Axes] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """ROC and precision-recall curves side by side."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes

    fpr, tpr, _ = roc_curve(y_true, y_score)
    m = classification_metrics(y_true, y_score)
    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={m['roc_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    base = float(np.mean(y_true))
    ax_pr.plot(rec, prec, label=f"{label} (AP={m['pr_auc']:.3f})")
    ax_pr.axhline(base, color="k", ls="--", alpha=0.4, label=f"base={base:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.legend(loc="upper right")
    return ax_roc, ax_pr


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
    bins: int = 40,
) -> plt.Axes:
    """Predicted-probability distribution split by actual class."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, density=True, label="good (0)")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, density=True, label="bad (1)")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution by class")
    ax.legend()
    return ax


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
    n_bins: int = 10,
    label: str = "model",
) -> plt.Axes:
    """Reliability curve vs the perfectly calibrated diagonal."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (reliability) curve")
    ax.legend(loc="upper left")
    return ax


def plot_confusion(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Confusion matrix at a chosen probability threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["good", "bad"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    return ax


def plot_learning_curve(
    estimator: Any,
    X: Any,
    y: Any,
    cv: Any,
    scoring: str = "roc_auc",
    train_sizes: np.ndarray | None = None,
    n_jobs: int = -1,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Train vs validation score as a function of training-set size."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=train_sizes, n_jobs=n_jobs, shuffle=True, random_state=42,
    )
    tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
    va_mean, va_std = val_scores.mean(1), val_scores.std(1)
    ax.plot(sizes, tr_mean, "o-", color="steelblue", label="train")
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")
    ax.plot(sizes, va_mean, "o-", color="darkorange", label="validation")
    ax.fill_between(sizes, va_mean - va_std, va_mean + va_std, alpha=0.15, color="darkorange")
    ax.set_xlabel("Training examples")
    ax.set_ylabel(scoring)
    ax.set_title("Learning curve")
    ax.legend(loc="best")
    return ax


def plot_validation_curve(
    estimator: Any,
    X: Any,
    y: Any,
    param_name: str,
    param_range: Sequence[Any],
    cv: Any,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    semilog: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Train vs validation score across one hyperparameter (bias/variance view)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=list(param_range),
        cv=cv, scoring=scoring, n_jobs=n_jobs,
    )
    tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
    va_mean, va_std = val_scores.mean(1), val_scores.std(1)
    x = list(param_range)
    plot = ax.semilogx if semilog else ax.plot
    plot(x, tr_mean, "o-", color="steelblue", label="train")
    plot(x, va_mean, "o-", color="darkorange", label="validation")
    ax.fill_between(x, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="steelblue")
    ax.fill_between(x, va_mean - va_std, va_mean + va_std, alpha=0.15, color="darkorange")
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(f"Validation curve over {param_name}")
    ax.legend(loc="best")
    return ax


def plot_loss_curve(
    train_loss: Sequence[float],
    valid_loss: Sequence[float],
    ax: plt.Axes | None = None,
    title: str = "Loss curve",
    ylabel: str = "log loss",
) -> plt.Axes:
    """Per-iteration train vs validation loss (boosting / SGD epochs)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    iters = np.arange(1, len(train_loss) + 1)
    ax.plot(iters, train_loss, color="steelblue", label="train")
    ax.plot(iters, valid_loss, color="darkorange", label="validation")
    best = int(np.argmin(valid_loss)) + 1
    ax.axvline(best, color="green", ls="--", alpha=0.6, label=f"best iter={best}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_param_loss_curve(
    values: Sequence[Any],
    train_loss: Sequence[float],
    valid_loss: Sequence[float],
    xlabel: str,
    ax: plt.Axes | None = None,
    title: str = "Train vs validation loss",
    semilog: bool = False,
) -> plt.Axes:
    """Train vs validation log loss across a complexity parameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    plot = ax.semilogx if semilog else ax.plot
    plot(list(values), train_loss, "o-", color="steelblue", label="train")
    plot(list(values), valid_loss, "o-", color="darkorange", label="validation")
    best = int(np.argmin(valid_loss))
    ax.axvline(list(values)[best], color="green", ls="--", alpha=0.6,
               label=f"best {xlabel}={list(values)[best]}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("log loss")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_feature_importance(
    names: Sequence[str],
    importances: Sequence[float],
    top: int = 20,
    ax: plt.Axes | None = None,
    title: str = "Feature importance",
) -> plt.Axes:
    """Horizontal bar chart of the top-N most important features."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    order = np.argsort(importances)[::-1][:top]
    names_s = [names[i] for i in order][::-1]
    vals_s = [importances[i] for i in order][::-1]
    ax.barh(names_s, vals_s, color="teal")
    ax.set_xlabel("importance")
    ax.set_title(title)
    return ax
