"""Lightweight model registry: persist fitted models and their metrics.

Models are stored as joblib files under `models/`, with a JSON index
(`models/model_registry.json`) holding holdout metrics and metadata so the
comparison notebook (and later the serving layer) can load them without
retraining.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import MODELS_DIR

REGISTRY_PATH: Path = MODELS_DIR / "model_registry.json"


def _model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.joblib"


def _read_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="ascii"))
    return {}


def _write_registry(reg: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2) + "\n", encoding="ascii")


def save_model(
    name: str,
    model: Any,
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted model and record its metrics in the registry index."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_path(name)
    joblib.dump(model, path)

    reg = _read_registry()
    reg[name] = {
        "path": path.name,
        "metrics": metrics or {},
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    _write_registry(reg)
    return path


def has_model(name: str) -> bool:
    """True when a persisted model file exists for this name."""
    return _model_path(name).exists()


def load_model(name: str) -> Any:
    """Load a persisted model by name."""
    path = _model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No saved model '{name}' at {path}")
    return joblib.load(path)


def list_models() -> list[str]:
    """Names of all models present in the registry index."""
    return sorted(_read_registry().keys())


def load_all_models() -> dict[str, Any]:
    """Load every model listed in the registry index."""
    return {name: load_model(name) for name in list_models() if has_model(name)}


def registry_metrics() -> dict[str, dict[str, float]]:
    """Return {name: metrics} for all registered models."""
    reg = _read_registry()
    return {name: info.get("metrics", {}) for name, info in reg.items()}
