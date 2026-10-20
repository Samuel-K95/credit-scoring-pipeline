# Credit scoring pipeline targets.
# Requires GNU Make and a Python 3.11 environment with requirements installed.

PYTHON ?= python
PIP ?= pip

.PHONY: help install data features train evaluate monitor serve test lint clean

help:
	@echo "Targets:"
	@echo "  install   Install Python dependencies"
	@echo "  data      Placeholder for raw data checks / download notes"
	@echo "  features  Build training features via DuckDB SQL"
	@echo "  train     Train scorecard and LightGBM models"
	@echo "  evaluate  Compute metrics and write reports/metrics.json"
	@echo "  monitor   Run PSI/CSI drift checks"
	@echo "  serve     Start FastAPI scoring service"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"
	@echo "  clean     Remove generated artifacts"

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -c "from pathlib import Path; from src.config import RAW_DATA_DIR, RAW_TABLE_FILES; missing=[f for f in RAW_TABLE_FILES if not (RAW_DATA_DIR/f).exists()]; print('Raw data dir:', RAW_DATA_DIR); print('Missing tables:', missing if missing else 'none')"

features:
	$(PYTHON) -m src.build_features

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

monitor:
	$(PYTHON) -m src.monitor

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check src api tests

clean:
	$(PYTHON) -c "import shutil; from pathlib import Path; from src import config as c; \
paths=[c.PROCESSED_DATA_DIR, c.MODELS_DIR, c.FIGURES_DIR]; \
[shutil.rmtree(p, ignore_errors=True) for p in paths]; \
c.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True); \
c.MODELS_DIR.mkdir(parents=True, exist_ok=True); \
c.FIGURES_DIR.mkdir(parents=True, exist_ok=True); \
print('Cleaned processed data, models, and figures.')"
