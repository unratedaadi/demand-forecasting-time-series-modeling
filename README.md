# Demand Forecasting / Time Series Modeling

A complete, dependency-free portfolio project that forecasts daily retail demand from simulated historical sales. It demonstrates a practical workflow: data generation, train/test split, benchmark comparison, forecast evaluation, output creation, and a lightweight dashboard.

## What it includes

- **Simulated retail data:** 540 daily observations with trend, weekly seasonality, monthly patterns, promotion uplift, and noise.
- **Models:** seasonal-naive benchmark and an interpretable trend-and-calendar regression.
- **Evaluation:** MAE, RMSE, and MAPE on a 90-day holdout period.
- **Outputs:** CSV forecast, model scorecard, SVG charts, and a standalone HTML dashboard.

## Run it

```bash
python3 run_pipeline.py
python3 -m unittest discover -s tests -v
```

Open `outputs/dashboard.html` in a browser after running the pipeline.

## Project structure

```text
data/                 Generated simulated source data
outputs/              Forecast, metrics, charts, and dashboard
src/forecasting.py    Forecasting and evaluation utilities
tests/                Standard-library unit tests
run_pipeline.py       End-to-end pipeline entry point
```

## Business interpretation

The selected model is chosen by lowest holdout MAE. In a real implementation, replace `data/synthetic_demand.csv` with validated sales history and add causal drivers such as price, stock availability, holidays, and confirmed promotion calendars. The current project intentionally labels all results as **simulated** so that it is suitable for teaching and portfolio use without overstating business validity.
