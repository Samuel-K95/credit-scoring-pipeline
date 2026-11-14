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

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_VERSION,
    REPORTS_DIR,
)
from src.modeling import (
    classification_metrics,
    evaluate_split,
    load_modeling_data,
    load_model,
)
from src.modeling import plots
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


def _write_model_card(metrics: dict, meta: dict, threshold: float) -> None:
    fam = meta.get("family", "unknown")
    src_model = meta.get("source_model", "champion")
    ho = metrics["holdout"]
    lines = [
        "# Model card -- credit scoring champion",
        "",
        f"- Model: `{src_model}` ({fam})",
        f"- Version: {MODEL_VERSION}",
        f"- Target: probability of default (TARGET = 1)",
        f"- Decision threshold (max-F1 on holdout): {threshold:.2f}",
        "",
        "## Holdout performance",
        "",
        f"- ROC AUC: {ho['roc_auc']:.4f}",
        f"- Gini: {ho['gini']:.4f}",
        f"- KS: {ho['ks']:.4f}",
        f"- PR AUC: {ho['pr_auc']:.4f}",
        f"- Log loss: {ho['log_loss']:.4f}",
        f"- Brier: {ho['brier']:.4f}",
        "",
        "## Notes",
        "",
        "- Preprocessing (median impute, p01/p99 cap, missing flags) is fit on",
        "  the training partition only and bundled inside the model pipeline.",
        "- Metrics are computed on a stratified holdout never used in tuning.",
        "- Home Credit uses relative day offsets, so no true out-of-time test is",
        "  possible; validation is a stratified holdout plus cross-validation.",
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

    proba_ho = model.predict_proba(data.X_holdout)[:, 1]
    figures = _save_figures(data.y_holdout.to_numpy(), proba_ho)

    payload = {
        "model": meta.get("source_model", "champion"),
        "family": meta.get("family"),
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(data.y_train)),
        "n_holdout": int(len(data.y_holdout)),
        "decision_threshold": round(threshold, 4),
        "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
        "holdout": {k: round(float(v), 6) for k, v in holdout_metrics.items()},
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
