# Credit scoring pipeline targets.
# Requires GNU Make and a Python 3.11 environment with requirements installed.

PYTHON ?= python
PIP ?= pip

.PHONY: help install data quality features preprocess train evaluate explain monitor serve test lint clean

help:
	@echo "Targets:"
	@echo "  install    Install Python dependencies"
	@echo "  data       Download/unzip Home Credit raw tables into data/raw"
	@echo "  quality    Run data quality cleaning and write reports/data_quality_report.md"
	@echo "  features   Build training features via DuckDB SQL"
	@echo "  preprocess Stratified splits + train-only impute/cap pipeline"
	@echo "  train      Train the champion model on the full training split"
	@echo "  evaluate  Compute metrics and write reports/metrics.json"
	@echo "  explain   SHAP global/local explanations into reports/"
	@echo "  monitor   Run PSI/CSI drift checks"
	@echo "  serve     Start FastAPI scoring service"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"
	@echo "  clean     Remove generated artifacts"

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) -m src.download_data

quality:
	$(PYTHON) -m src.data_quality

features:
	$(PYTHON) -m src.features

preprocess:
	$(PYTHON) -m src.preprocess

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

explain:
	$(PYTHON) -m src.explain

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
