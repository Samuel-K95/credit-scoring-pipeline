"""Explain the trained champion with SHAP and write report artifacts.

Produces:
- reports/figures/shap_global_bar.png     mean absolute SHAP per feature
- reports/figures/shap_beeswarm.png        signed SHAP spread (top features)
- reports/figures/shap_dependence_*.png    value vs contribution, top drivers
- reports/figures/shap_waterfall_*.png     local explanations for sample cases
- reports/explainability.md                ranked global drivers + notes

Usage:
    python -m src.explain
    python -m src.explain --sample 4000 --top 20
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXPLAINABILITY_REPORT_PATH,
    FIGURES_DIR,
    MODEL_VERSION,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src.modeling import compute_shap, load_model, load_modeling_data
from src.modeling import explain as ex
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


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def run_explain(sample: int = 4000, top: int = 20, n_dependence: int = 4) -> dict:
    """Compute SHAP on a holdout sample and write figures plus a markdown report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_champion()
    meta = _champion_meta()
    data = load_modeling_data()

    n = min(sample, len(data.X_holdout))
    X = data.X_holdout.sample(n=n, random_state=RANDOM_SEED)
    y = data.y_holdout.loc[X.index]
    result = compute_shap(model, X)
    importance = result.global_importance()

    saved: list[str] = []

    fig, ax = plt.subplots(figsize=(8, 8))
    ex.plot_global_bar(result, ax=ax, top=top)
    p = FIGURES_DIR / "shap_global_bar.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    fig, ax = plt.subplots(figsize=(9, 8))
    ex.plot_beeswarm(result, ax=ax, top=min(top, 15))
    p = FIGURES_DIR / "shap_beeswarm.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(p.name)

    for feat in list(importance.head(n_dependence).index):
        fig, ax = plt.subplots(figsize=(6, 5))
        ex.plot_dependence(result, feat, ax=ax)
        p = FIGURES_DIR / f"shap_dependence_{_sanitize(feat)}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    # Local explanations: highest and lowest predicted-risk applicants.
    proba = model.predict_proba(X)[:, 1]
    cases = {
        "highest_risk": int(np.argmax(proba)),
        "lowest_risk": int(np.argmin(proba)),
    }
    for label, row in cases.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        ex.plot_waterfall(result, row=row, ax=ax, top=12)
        p = FIGURES_DIR / f"shap_waterfall_{label}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(p.name)

    _write_report(importance, meta, result, n, int(y.sum()), saved, top)

    summary = {
        "model": meta.get("source_model", "champion"),
        "n_explained": n,
        "top_features": list(importance.head(10).index),
        "figures": saved,
        "report": str(EXPLAINABILITY_REPORT_PATH),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _write_report(importance, meta, result, n, n_pos, figures, top) -> None:
    fam = meta.get("family", "boosting")
    src_model = meta.get("source_model", "champion")
    base_prob = 1.0 / (1.0 + np.exp(-result.base_value))
    lines = [
        "# Explainability -- credit scoring champion",
        "",
        f"- Model: `{src_model}` ({fam}), version {MODEL_VERSION}",
        f"- Rows explained: {n} holdout applicants ({n_pos} defaults)",
        f"- SHAP space: margin (log-odds); base value {result.base_value:.4f}"
        f" (PD {base_prob:.4f})",
        "",
        f"## Top {top} global drivers (mean absolute SHAP)",
        "",
        "| Rank | Feature | mean(|SHAP|) |",
        "| ---- | ------- | ------------ |",
    ]
    for i, (feat, val) in enumerate(importance.head(top).items(), start=1):
        lines.append(f"| {i} | `{feat}` | {val:.4f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- SHAP values are exact tree contributions from the booster, so they",
        "  reconstruct the raw score (`sum(SHAP) + base = model margin`).",
        "- Preprocessing (median impute, p01/p99 cap, missing flags) runs before",
        "  SHAP, so contributions are attributed to the model-ready features.",
        "- Positive SHAP pushes predicted default risk up; negative pushes it down.",
        "",
        "## Figures",
        "",
    ]
    for name in figures:
        lines.append(f"- `reports/figures/{name}`")
    lines.append("")
    EXPLAINABILITY_REPORT_PATH.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP explainability for the champion.")
    parser.add_argument("--sample", type=int, default=4000, help="Holdout rows to explain.")
    parser.add_argument("--top", type=int, default=20, help="Top features to report.")
    parser.add_argument(
        "--dependence", type=int, default=4, help="Dependence plots for top-N drivers."
    )
    args = parser.parse_args()
    run_explain(sample=args.sample, top=args.top, n_dependence=args.dependence)


if __name__ == "__main__":
    main()
