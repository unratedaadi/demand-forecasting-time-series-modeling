# Demand Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free, reproducible retail demand-forecasting portfolio project.

**Architecture:** `src/forecasting.py` owns data creation, features, models, metrics, exports, and presentation helpers. `run_pipeline.py` orchestrates a deterministic train/test experiment and creates dashboard-ready assets. Standard-library CSV and SVG keep the project runnable in the available environment.

**Tech Stack:** Python 3 standard library, CSV, SVG, HTML, unittest.

**Spec:** `docs/superpowers/specs/2026-10-24-demand-forecasting-design.md`

## Global Constraints

- Use no non-standard Python packages.
- Seed generated data for reproducibility.
- Label every result as simulated demand.
- Run the workflow with `python3 run_pipeline.py`.

## Review Focus

- A 7-day forecast should begin the day after the latest historical date.
- Forecast values must never be negative.
- MAPE must ignore zero actual values.
- Feature generation must use only prior demand values for lags.
- The report must cite UCI Sales Transactions Weekly and disclose its relative-week time index.

### Task 1: Forecasting core

**Files:**
- Create: `src/forecasting.py`, `src/__init__.py`
- Modify: `tests/test_forecasting.py`

**Interfaces:**
- Produces: `aggregate_weekly_sales`, `seasonal_naive_forecast`, `fit_linear_model`, `recursive_linear_forecast`, `evaluate_forecast`.

- [ ] Write dependency-free failing tests for feature dates, forecast horizon, and MAE/RMSE/MAPE.
- [ ] Run `python3 -m unittest discover -s tests -v` and confirm failure before implementation.
- [ ] Implement the forecasting core using only the standard library.
- [ ] Run the test suite and confirm it passes.

### Task 2: Reproducible pipeline and report

**Files:**
- Create: `run_pipeline.py`, `scripts/prepare_uci_weekly_sales.py`, `README.md`
- Produces: `data/uci_weekly_sales_demand.csv`, `outputs/forecast.csv`, `outputs/model_metrics.csv`, `outputs/dashboard.html`, `outputs/demand_history.svg`, `outputs/forecast_comparison.svg`.

- [ ] Orchestrate train/test evaluation and select the lowest-MAE model.
- [ ] Export forecast and metrics artifacts plus readable SVG/HTML summaries.
- [ ] Run `python3 run_pipeline.py` and inspect generated outputs.

### Task 3: Final verification

**Files:**
- Modify: `tests/test_forecasting.py` if additional edge coverage is needed.

- [ ] Verify output dates, non-negative forecasts, and simulated-data labels.
- [ ] Run the full unit suite and pipeline from a clean output directory.
