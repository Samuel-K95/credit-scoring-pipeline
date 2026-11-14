"""Evaluate the trained champion and write the official metrics artifacts.

Produces:
- reports/metrics.json   machine-readable train/holdout metrics (source of truth)
- reports/model_card.md  short human-readable summary
- reports/figures/*.png  ROC, calibration, and score-distribution plots

Usage:
    python -m src.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_VERSION,
    PSI_STABLE_MAX,
    PSI_WATCH_MAX,
    REPORTS_DIR,
)
from src.modeling import (
    cutoff_table,
    evaluate_split,
    grouped_discrimination,
    load_modeling_data,
    load_model,
)
from src.modeling import plots
from src.modeling.metrics import DEFAULT_LGD
from src.modeling.registry import REGISTRY_PATH


def _load_champion():
    try:
        return load_model("champion")
    except FileNotFoundError as exc:
        raise SystemExit("No champion model found. Run: python -m src.train") from exc


def _champion_meta() -> dict:
    if REGISTRY_PATH.exists():
        reg = json.loads(REGISTRY_PATH.read_text(encoding="ascii"))
        return reg.get("champion", {})
    return {}


def _save_figures(y_true, proba) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plots.plot_roc_pr(y_true, proba, label="champion", axes=axes)
    p = FIGURES_DIR / "champion_roc_pr.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    fig, ax = plt.subplots(figsize=(6, 6))
    plots.plot_calibration(y_true, proba, ax=ax, label="champion")
    p = FIGURES_DIR / "champion_calibration.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    fig, ax = plt.subplots(figsize=(8, 5))
    plots.plot_score_distribution(y_true, proba, ax=ax)
    p = FIGURES_DIR / "champion_score_distribution.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)
    return saved


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=[-np.inf, 30, 40, 50, np.inf],
        labels=["under_30", "30_to_40", "40_to_50", "50_plus"],
    ).astype(str)


def _income_band(train_income: pd.Series, holdout_income: pd.Series) -> pd.Series:
    """Quartile labels from the training income distribution, applied to holdout."""
    edges = [
        float(v)
        for v in train_income.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    ]
    cuts = [edges[0]]
    for edge in edges[1:]:
        if edge > cuts[-1]:
            cuts.append(edge)
    bins = [-np.inf, *cuts, np.inf]
    labels = ["income_q1", "income_q2", "income_q3", "income_q4"][: len(bins) - 1]
    return pd.cut(
        holdout_income, bins=bins, labels=labels, include_lowest=True
    ).astype(str)


_AGE_ORDER = ("under_30", "30_to_40", "40_to_50", "50_plus")
_INCOME_ORDER = ("income_q1", "income_q2", "income_q3", "income_q4")


def _sort_segments(rows: list[dict], order: tuple[str, ...]) -> list[dict]:
    rank = {name: i for i, name in enumerate(order)}
    return sorted(rows, key=lambda row: rank.get(str(row["segment"]), len(order)))


def _round_metric_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        cleaned: dict = {}
        for key, value in row.items():
            if isinstance(value, float):
                cleaned[key] = round(value, 6)
            else:
                cleaned[key] = value
        out.append(cleaned)
    return out


def _fmt_auc(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _write_model_card(metrics: dict, meta: dict, threshold: float) -> None:
    fam = meta.get("family", "unknown")
    src_model = meta.get("source_model", "champion")
    ho = metrics["holdout"]
    lines = [
        "# Model card -- credit scoring champion",
        "",
        f"- Model: `{src_model}` ({fam})",
        f"- Version: {MODEL_VERSION}",
        "- Target: payment difficulties (TARGET = 1) on Home Credit application data",
        "- Output: an uncalibrated risk score. `scale_pos_weight` was used for",
        "  the 8 percent default rate, so the number is not a calibrated PD.",
        f"- Decision threshold (max-F1 on holdout): {threshold:.2f}",
        "- Intended use: rank applicants and support approve / refer / decline.",
        "  Not a sole automated decision, and not a priced expected-loss model.",
        "",
        "## Training data",
        "",
        f"- Train rows: {metrics['n_train']}",
        f"- Holdout rows: {metrics['n_holdout']}",
        "- Features: application ratios and external scores, plus bureau,",
        "  previous-application, instalment, POS, and credit-card aggregates.",
        "- Gender, family status, and other protected attributes are not inputs.",
        "",
        "## Holdout performance",
        "",
        f"- ROC AUC: {ho['roc_auc']:.4f}",
        f"- Gini: {ho['gini']:.4f}",
        f"- KS: {ho['ks']:.4f}",
        f"- PR AUC: {ho['pr_auc']:.4f} (more informative than ROC under imbalance)",
        f"- Log loss: {ho['log_loss']:.4f}",
        f"- Brier: {ho['brier']:.4f}",
        "",
        "## Cutoff analysis (illustrative)",
        "",
        "Approve when the risk score is below the threshold. The loss proxy is",
        "`approval_rate * bad_rate_approved * LGD` with exposure 1 and LGD 1.",
        "It ignores interest income and is not a capital or pricing number.",
        "",
        "| Threshold | Approval rate | Bad rate approved | Loss proxy |",
        "| --------- | ------------- | ----------------- | ---------- |",
    ]
    for row in metrics["cutoff"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['approval_rate']:.4f} "
            f"| {row['bad_rate_approved']:.4f} | {row['expected_loss_proxy']:.4f} |"
        )
    lines += [
        "",
        "## Segment stability (holdout ROC AUC)",
        "",
        "Income bands use training-set quartiles. A missing AUC means the",
        "segment had only one class.",
        "",
        "### Age",
        "",
        "| Segment | n | Default rate | ROC AUC |",
        "| ------- | - | ------------ | ------- |",
    ]
    for row in metrics["segments"]["age"]:
        lines.append(
            f"| {row['segment']} | {row['n']} | {row['default_rate']:.4f} "
            f"| {_fmt_auc(row['roc_auc'])} |"
        )
    lines += [
        "",
        "### Income",
        "",
        "| Segment | n | Default rate | ROC AUC |",
        "| ------- | - | ------------ | ------- |",
    ]
    for row in metrics["segments"]["income"]:
        lines.append(
            f"| {row['segment']} | {row['n']} | {row['default_rate']:.4f} "
            f"| {_fmt_auc(row['roc_auc'])} |"
        )
    lines += [
        "",
        "## Fairness",
        "",
        "- Protected attributes from the source application table are not model",
        "  features. Age and family size are used and can correlate with",
        "  protected groups.",
        "- No disparate-impact audit was run. Review segment stability above",
        "  before using the score in a live decision policy.",
        "",
        "## Monitoring",
        "",
        "- Score PSI below "
        f"{PSI_STABLE_MAX:.2f} is stable, {PSI_STABLE_MAX:.2f} to "
        f"{PSI_WATCH_MAX:.2f} is watch, and {PSI_WATCH_MAX:.2f} or above is alert.",
        "- Retrain or investigate when the score PSI is in alert, or when two or",
        "  more of the top drivers are in alert on the same comparison window.",
        "- Home Credit has no calendar dates, so the shipped comparison is",
        "  train versus holdout. A shifted sample in notebook 12 shows the alert",
        "  path; label any such run as simulated.",
        "",
        "## Known risks",
        "",
        "- The max-F1 threshold is chosen on the holdout, so that operating point",
        "  is optimistic. ROC AUC, Gini, KS, and PR AUC do not use it.",
        "- Scores are not calibrated. Do not treat them as probabilities in a",
        "  pricing or IFRS-style expected-loss formula.",
        "- Validation is a stratified holdout plus cross-validation. A true",
        "  out-of-time back-test is not possible on these relative day offsets.",
        "",
    ]
    MODEL_CARD_PATH.write_text("\n".join(lines), encoding="ascii")


def run_evaluate() -> dict:
    """Load the champion, compute metrics, and write reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_champion()
    meta = _champion_meta()
    data = load_modeling_data()

    train_metrics = evaluate_split(model, data.X_train, data.y_train)
    holdout_metrics = evaluate_split(model, data.X_holdout, data.y_holdout)
    threshold = float(holdout_metrics["best_f1_threshold"])

    y_ho = data.y_holdout.to_numpy()
    proba_ho = model.predict_proba(data.X_holdout)[:, 1]
    figures = _save_figures(y_ho, proba_ho)
    cutoffs = _round_metric_rows(cutoff_table(y_ho, proba_ho, lgd=DEFAULT_LGD))
    age_rows = _sort_segments(
        _round_metric_rows(
            grouped_discrimination(
                y_ho, proba_ho, _age_band(data.X_holdout["app_age_years"]).to_numpy()
            )
        ),
        _AGE_ORDER,
    )
    income_rows = _sort_segments(
        _round_metric_rows(
            grouped_discrimination(
                y_ho,
                proba_ho,
                _income_band(
                    data.X_train["app_amt_income_total"],
                    data.X_holdout["app_amt_income_total"],
                ).to_numpy(),
            )
        ),
        _INCOME_ORDER,
    )

    payload = {
        "model": meta.get("source_model", "champion"),
        "family": meta.get("family"),
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(data.y_train)),
        "n_holdout": int(len(data.y_holdout)),
        "decision_threshold": round(threshold, 4),
        "lgd_assumption": DEFAULT_LGD,
        "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
        "holdout": {k: round(float(v), 6) for k, v in holdout_metrics.items()},
        "cutoff": cutoffs,
        "segments": {"age": age_rows, "income": income_rows},
        "figures": figures,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="ascii")
    _write_model_card(payload, meta, threshold)

    print(json.dumps(
        {
            "model": payload["model"],
            "holdout_roc_auc": payload["holdout"]["roc_auc"],
            "holdout_ks": payload["holdout"]["ks"],
            "metrics_path": str(METRICS_PATH),
            "model_card": str(MODEL_CARD_PATH),
        },
        indent=2,
    ), flush=True)
    return payload


def main() -> None:
    run_evaluate()


if __name__ == "__main__":
    main()
