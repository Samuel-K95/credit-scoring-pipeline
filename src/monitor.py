"""PSI / CSI drift monitoring between a reference and a comparison set.

Population Stability Index (PSI) measures how much a distribution has shifted
relative to a reference. Applied to the model output it flags score drift;
applied per feature (Characteristic Stability Index, CSI) it flags input drift.

Home Credit uses relative day offsets, so there is no true out-of-time sample.
The runnable baseline therefore treats the training partition as the reference
and the holdout partition as the current population; stable PSI confirms the
holdout matches training. The companion notebook simulates a shifted population
to show the index reacting.

Bands follow common practice (configurable in `src.config`):
- PSI  < 0.10            stable
- 0.10 <= PSI < 0.25     watch
- PSI >= 0.25            alert
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DRIFT_JSON_PATH,
    DRIFT_REPORT_PATH,
    MODEL_VERSION,
    PSI_STABLE_MAX,
    PSI_WATCH_MAX,
    REPORTS_DIR,
)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """PSI of `actual` versus `expected` using expected-quantile bins.

    NaNs are dropped before binning. Low-cardinality inputs that cannot form
    at least two bins return 0.0 (nothing to compare).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0].astype(float)
    a_counts = np.histogram(actual, bins=edges)[0].astype(float)
    eps = 1e-6
    e_pct = np.clip(e_counts / e_counts.sum(), eps, None)
    a_pct = np.clip(a_counts / a_counts.sum(), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_status(psi: float) -> str:
    """Classify a PSI value into stable / watch / alert."""
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_WATCH_MAX:
        return "watch"
    return "alert"


def feature_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Per-feature CSI table sorted from most to least drift."""
    cols = columns or [c for c in reference.columns if c in current.columns]
    rows = []
    for col in cols:
        psi = population_stability_index(
            reference[col].to_numpy(), current[col].to_numpy(), bins=bins
        )
        rows.append({"feature": col, "psi": psi, "status": drift_status(psi)})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return out


def score_psi(
    model: Any, reference_X: pd.DataFrame, current_X: pd.DataFrame, bins: int = 10
) -> float:
    """PSI of the model's risk score between reference and current inputs."""
    ref = model.predict_proba(reference_X)[:, 1]
    cur = model.predict_proba(current_X)[:, 1]
    return population_stability_index(ref, cur, bins=bins)


def _status_counts(table: pd.DataFrame) -> dict[str, int]:
    counts = table["status"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in ("stable", "watch", "alert")}


def run_monitor(bins: int = 10, top: int = 20) -> dict[str, Any]:
    """Compute score + feature drift (train vs holdout) and write reports."""
    from src.modeling import load_model, load_modeling_data

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model("champion")
    data = load_modeling_data()

    s_psi = score_psi(model, data.X_train, data.X_holdout, bins=bins)
    feat = feature_psi(data.X_train, data.X_holdout, data.feature_cols, bins=bins)
    counts = _status_counts(feat)

    payload = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": "train",
        "current": "holdout",
        "n_reference": int(len(data.X_train)),
        "n_current": int(len(data.X_holdout)),
        "bins": bins,
        "score_psi": round(s_psi, 6),
        "score_status": drift_status(s_psi),
        "feature_status_counts": counts,
        "features": [
            {"feature": r.feature, "psi": round(float(r.psi), 6), "status": r.status}
            for r in feat.itertuples(index=False)
        ],
    }
    DRIFT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="ascii")
    _write_report(payload, feat, top)

    print(json.dumps(
        {
            "score_psi": payload["score_psi"],
            "score_status": payload["score_status"],
            "feature_status_counts": counts,
            "report": str(DRIFT_REPORT_PATH),
        },
        indent=2,
    ), flush=True)
    return payload


def _write_report(payload: dict[str, Any], feat: pd.DataFrame, top: int) -> None:
    counts = payload["feature_status_counts"]
    lines = [
        "# Drift monitoring report",
        "",
        f"- Model version: {payload['model_version']}",
        f"- Reference: {payload['reference']} (n={payload['n_reference']})",
        f"- Current: {payload['current']} (n={payload['n_current']})",
        f"- PSI bins: {payload['bins']}",
        "",
        "## Score drift (model output)",
        "",
        f"- Score PSI: {payload['score_psi']:.4f} ({payload['score_status']})",
        "",
        "## Feature drift (CSI)",
        "",
        f"- stable: {counts['stable']}, watch: {counts['watch']},"
        f" alert: {counts['alert']}",
        "",
        f"### Top {top} features by PSI",
        "",
        "| Feature | PSI | Status |",
        "| ------- | --- | ------ |",
    ]
    for r in feat.head(top).itertuples(index=False):
        lines.append(f"| `{r.feature}` | {r.psi:.4f} | {r.status} |")
    lines += [
        "",
        "## Notes",
        "",
        "- PSI uses reference-quantile bins (default 10); NaNs are dropped per",
        "  column before binning.",
        "- Bands: stable < 0.10, watch 0.10-0.25, alert >= 0.25.",
        "- The train-vs-holdout baseline is expected to be stable; see",
        "  `notebooks/12_monitoring_drift.ipynb` for a simulated drift scenario.",
        "",
    ]
    DRIFT_REPORT_PATH.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    run_monitor()


if __name__ == "__main__":
    main()
