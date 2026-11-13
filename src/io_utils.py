"""Small shared I/O helpers used across Stage 1-4 modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CSV_CHUNKSIZE, DAYS_EMPLOYED_ANOMALY


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with ASCII encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="ascii", errors="replace")


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table without requiring the tabulate package."""
    if df.empty:
        return "_empty_"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def count_csv_rows(path: Path) -> int:
    """Count data rows (excludes header) without parsing CSV fields."""
    with path.open("rb") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def unique_ids(path: Path, id_col: str) -> set[int]:
    """Collect unique integer IDs from a CSV in chunks."""
    ids: set[int] = set()
    for chunk in pd.read_csv(path, usecols=[id_col], chunksize=CSV_CHUNKSIZE):
        vals = chunk[id_col].dropna().astype("int64").unique()
        ids.update(vals.tolist())
    return ids


def replace_sentinel(
    series: pd.Series,
    sentinel: int = DAYS_EMPLOYED_ANOMALY,
) -> tuple[pd.Series, pd.Series]:
    """Replace a known sentinel with NaN and return a binary anomaly flag."""
    flag = (series == sentinel).astype("int8")
    cleaned = series.mask(series == sentinel, np.nan)
    return cleaned, flag
