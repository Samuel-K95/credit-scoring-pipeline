"""Train the champion PD model and persist it for evaluation and serving.

The champion family and its tuned hyperparameters come from the Stage-5
notebooks, captured in `src.modeling.champions`. The selected family is read
from `reports/model_comparison.json` when present (written by notebook 10),
otherwise it falls back to the known best (`lightgbm`).

Usage:
    python -m src.train                 # train the selected champion
    python -m src.train --model xgboost # train a specific family champion
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from src.config import MODEL_COMPARISON_PATH, MODEL_VERSION
from src.modeling import (
    build_champion,
    champion_names,
    evaluate_split,
    load_modeling_data,
    make_pipeline,
    save_model,
)
from src.modeling.champions import FAMILY

DEFAULT_CHAMPION = "lightgbm"


def resolve_champion_name(explicit: str | None = None) -> str:
    """Pick the champion family: explicit arg, comparison report, or default."""
    if explicit:
        if explicit not in champion_names():
            raise SystemExit(
                f"Unknown model '{explicit}'. Options: {champion_names()}"
            )
        return explicit
    if MODEL_COMPARISON_PATH.exists():
        try:
            data = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="ascii"))
            name = data.get("champion")
            if name in champion_names():
                return name
        except (ValueError, OSError):
            pass
    return DEFAULT_CHAMPION


def train_champion(name: str | None = None) -> dict[str, Any]:
    """Fit the champion pipeline on the full training partition and save it."""
    champion = resolve_champion_name(name)
    print(f"Training champion: {champion} ({FAMILY.get(champion)})", flush=True)

    data = load_modeling_data()
    pipe = make_pipeline(build_champion(champion, data.pos_weight))

    t0 = time.perf_counter()
    pipe.fit(data.X_train, data.y_train)
    fit_s = time.perf_counter() - t0

    holdout_metrics = evaluate_split(pipe, data.X_holdout, data.y_holdout)
    path = save_model(
        "champion",
        pipe,
        metrics=holdout_metrics,
        extra={
            "family": FAMILY.get(champion),
            "source_model": champion,
            "model_version": MODEL_VERSION,
            "n_train": int(len(data.X_train)),
            "feature_cols": data.feature_cols,
        },
    )

    summary = {
        "champion": champion,
        "family": FAMILY.get(champion),
        "model_version": MODEL_VERSION,
        "n_train": int(len(data.X_train)),
        "fit_seconds": round(fit_s, 1),
        "holdout_roc_auc": round(holdout_metrics["roc_auc"], 4),
        "holdout_ks": round(holdout_metrics["ks"], 4),
        "model_path": str(path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the champion PD model.")
    parser.add_argument(
        "--model",
        default=None,
        help="Champion family to train (default: from comparison report).",
    )
    args = parser.parse_args()
    train_champion(args.model)


if __name__ == "__main__":
    main()
