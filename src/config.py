"""Project paths, seeds, and shared constants."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

SQL_DIR: Path = PROJECT_ROOT / "sql"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

TRAIN_FEATURES_PATH: Path = PROCESSED_DATA_DIR / "train_features.parquet"
SPLIT_INDICES_PATH: Path = PROCESSED_DATA_DIR / "split_indices.json"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
DATA_QUALITY_REPORT_PATH: Path = REPORTS_DIR / "data_quality_report.md"
MODEL_CARD_PATH: Path = REPORTS_DIR / "model_card.md"
DRIFT_REPORT_PATH: Path = REPORTS_DIR / "drift_report.md"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Home Credit table names (files under data/raw/)
# ---------------------------------------------------------------------------

APPLICATION_TRAIN_FILE: str = "application_train.csv"
BUREAU_FILE: str = "bureau.csv"
BUREAU_BALANCE_FILE: str = "bureau_balance.csv"
PREVIOUS_APPLICATION_FILE: str = "previous_application.csv"
INSTALLMENTS_PAYMENTS_FILE: str = "installments_payments.csv"
POS_CASH_BALANCE_FILE: str = "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE_FILE: str = "credit_card_balance.csv"

RAW_TABLE_FILES: tuple[str, ...] = (
    APPLICATION_TRAIN_FILE,
    BUREAU_FILE,
    BUREAU_BALANCE_FILE,
    PREVIOUS_APPLICATION_FILE,
    INSTALLMENTS_PAYMENTS_FILE,
    POS_CASH_BALANCE_FILE,
    CREDIT_CARD_BALANCE_FILE,
)

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------

ID_COL: str = "SK_ID_CURR"
TARGET_COL: str = "TARGET"

# Known Home Credit sentinel: unemployed / missing employment coded as this.
DAYS_EMPLOYED_ANOMALY: int = 365243

# Feature column prefixes by source table family.
FEATURE_PREFIXES: tuple[str, ...] = (
    "inst_",
    "bur_",
    "prev_",
    "pos_",
    "cc_",
    "app_",
)

# ---------------------------------------------------------------------------
# Modelling / scorecard defaults
# ---------------------------------------------------------------------------

HOLDOUT_SIZE: float = 0.20
N_CV_FOLDS: int = 5

SCORECARD_BASE_SCORE: int = 600
SCORECARD_PDO: int = 50  # points to double the odds

MODEL_VERSION: str = "0.1.0"

# ---------------------------------------------------------------------------
# Drift monitoring thresholds (PSI)
# ---------------------------------------------------------------------------

PSI_STABLE_MAX: float = 0.10
PSI_WATCH_MAX: float = 0.25
