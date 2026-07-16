# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repo contains two independently packaged Python projects:
- `merlion/` — the core time series intelligence library (forecasting, anomaly detection, change point detection).
- `ts_datasets/` — a standalone sub-package (`ts_datasets/setup.py`) providing standardized data loaders that return `pandas.DataFrame`s with metadata. Must be installed separately, in editable mode (`-e`), to avoid manually specifying dataset root directories.

## Common Commands

Install for development (from repo root):
```shell script
pip install pre-commit && pre-commit install
pip install -e .[all]
pip install -e ts_datasets/
```

Run tests (pytest, configured in `pytest.ini` with live logging enabled):
```shell script
pytest tests/                          # full suite
pytest tests/anomaly/test_isolation_forest.py         # single file
pytest tests/anomaly/test_isolation_forest.py::test_name   # single test
```
Test modules mirror `merlion`'s structure: `tests/anomaly`, `tests/change_point`, `tests/evaluate`, `tests/forecast`, `tests/transform`, `tests/spark`.

Benchmark scripts (used to produce the results in the technical report):
```shell script
python benchmark_anomaly.py --dataset NAB_realAWSCloudwatch --model IsolationForest --retrain_freq 1d
python benchmark_forecast.py --dataset M4_Hourly --model ETS
```
Model/dataset configs for these scripts live in `conf/benchmark_anomaly.json` and `conf/benchmark_forecast.json`.

Formatting/license headers are enforced via pre-commit (`black --line-length 120`, `licenseheaders` using `.copyright.tmpl`) — run `pre-commit install` once so commits are auto-formatted.

Optional extras: `dashboard` (GUI, `python -m merlion.dashboard`, served at http://localhost:8050), `spark` (PySpark distributed backend), `deep-learning` (torch-based models). Some anomaly models need a JDK on `PATH`/`JAVA_HOME`; some forecasting models need OpenMP (`conda install -c conda-forge lightgbm` or `brew install libomp`).

## Architecture

### Model type hierarchy
All models are unified under a shared config/model interface, and fall into two families under `merlion/models/`:
- **Anomaly detectors** (`merlion/models/anomaly/`) inherit from `merlion.models.anomaly.base.DetectorBase`, with a paired `DetectorConfig`. Required overrides: `_train()` (returns anomaly scores on train data) and `_get_anomaly_score()`.
- **Forecasters** (`merlion/models/forecast/`) inherit from `merlion.models.forecast.base.ForecasterBase`, with a paired `ForecasterConfig` (must accept `max_forecast_steps` if the model has a bounded horizon). Required overrides: `_train()` and `_forecast()`.
- **Forecaster-based anomaly detectors** (`merlion/models/anomaly/forecast_based/`) convert a forecaster's residual into an anomaly score, via multiple inheritance from `ForecastingDetectorBase` and the forecaster class (in that order); configs likewise multiply-inherit `ForecasterConfig` and `DetectorConfig`. See `merlion/models/anomaly/forecast_based/prophet.py` for a worked example.

Every new model must be registered with the model factory (`merlion/models/factory.py`) to be constructible generically.

Config classes support optional class-variable overrides that plug into shared pipeline behavior instead of subclassing it:
- `ConfigClass._default_transform` — pre-processing transform applied if none is given (default: `Identity`).
- `ConfigClass._default_post_rule` — post-processing rule applied to anomaly scores (used to compute `model.get_anomaly_label()` from `model.get_anomaly_score()`).
- `ModelClass._default_post_rule_train_config` — default config for training that post-rule (e.g. detection threshold selection).

`merlion/models/automl/` and `merlion/models/ensemble/` build on top of these base models (AutoML tuning/selection, and combining multiple models' outputs), rather than being separate model families.

### Transforms
Data pre-processing transforms (`merlion/transform/`) inherit `TransformBase` or `InvertibleTransformBase`. Even non-invertible transforms must support a pseudo-inverse (e.g. `MovingAverage` approximates inversion via stored boundary values and de-convolution). Implement `train()` (no-op if stateless), `__call__()`, and `_invert()`, and set `requires_inversion_state` depending on whether inversion needs state carried from the forward pass (set via `self.inversion_state` in `__call__`). Register new transforms with the transform factory (`merlion/transform/factory.py`).

### Datasets
New data loaders go in `ts_datasets/ts_datasets/anomaly` (labeled anomalies) or `ts_datasets/ts_datasets/forecast` (no labels), and must be exported from that subpackage's `__init__.py` `__all__`. Raw data files live under `data/`.

### Evaluation pipeline
`merlion/evaluate/` simulates live production deployment: train on a historical window, retrain at a regular cadence (full history or a limited window), obtain predictions between retrainings (batch, streaming, or intermediate), then score against ground truth. `TSADMetric` (anomaly) and `ForecastMetric` (forecasting) provide the metrics (precision/recall/F1/MTTD; sMAPE/MSIS/etc.).

### Other components
- `merlion/spark/` — PySpark backend for running Merlion at scale.
- `merlion/dashboard/` — Dash-based GUI for interactively exploring models/datasets.
- `merlion/post_process/` — anomaly score post-processing rules (thresholds, calibration) referenced by `DetectorConfig._default_post_rule`.

## Adding a New Model or Transform (docs)

When adding a model/transform, in addition to code + tests, update the Sphinx docs: add the new module to the relevant `__init__.py` autosummary block (e.g. `merlion/models/anomaly/__init__.py`) and to the corresponding `docs/source/merlion.*.rst` file.
