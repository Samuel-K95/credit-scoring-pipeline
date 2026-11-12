"""Download Home Credit raw tables and print an inventory.

Ported from notebooks/01_download_and_explore.ipynb (non-interactive path).
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    KAGGLE_COMPETITION,
    RAW_DATA_DIR,
    RAW_TABLE_FILES,
)


def expected_csvs_present(raw_dir: Path = RAW_DATA_DIR) -> bool:
    """Return True when all modelling tables exist under raw_dir."""
    return all((raw_dir / name).exists() for name in RAW_TABLE_FILES)


def download_competition(raw_dir: Path = RAW_DATA_DIR) -> Path:
    """Download the Kaggle competition zip into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_COMPETITION,
        "-p",
        str(raw_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    zip_path = raw_dir / f"{KAGGLE_COMPETITION}.zip"
    return zip_path


def unzip_archives(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Extract all zip archives found under raw_dir."""
    extracted: list[Path] = []
    for zpath in sorted(raw_dir.glob("*.zip")):
        print(f"Extracting {zpath.name} ...")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(raw_dir)
        extracted.append(zpath)
    return extracted


def inventory_raw(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """List files under raw_dir with sizes."""
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.iterdir()):
        if path.name == ".gitkeep":
            continue
        rows.append(
            {
                "file": path.name,
                "type": path.suffix.lower() or "dir",
                "size_mb": round(path.stat().st_size / (1024**2), 2),
                "expected_table": path.name in RAW_TABLE_FILES,
            }
        )
    return pd.DataFrame(rows).sort_values("size_mb", ascending=False)


def ensure_raw_data(raw_dir: Path = RAW_DATA_DIR) -> None:
    """Download and unzip if expected CSVs are missing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if expected_csvs_present(raw_dir):
        print("All expected CSVs already present. Skipping download.")
        return
    if not any(raw_dir.glob("*.zip")):
        download_competition(raw_dir)
    unzip_archives(raw_dir)
    missing = [f for f in RAW_TABLE_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected tables after download/extract: "
            + ", ".join(missing)
            + ". If you saw HTTP 403, accept competition rules at "
            f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules"
        )


def main() -> None:
    """CLI entry: ensure raw data, then print inventory and missing check."""
    ensure_raw_data()
    inv = inventory_raw()
    print("\nRaw data inventory:")
    print(inv.to_string(index=False))
    missing = [f for f in RAW_TABLE_FILES if not (RAW_DATA_DIR / f).exists()]
    print("\nMissing expected tables:" if missing else "\nAll expected tables present.")
    if missing:
        print(missing)


if __name__ == "__main__":
    main()
