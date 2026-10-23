# Demand Forecasting / Time Series Modeling

A complete, dependency-free portfolio project that forecasts weekly retail demand from real historical product-purchase data. It demonstrates a practical workflow: source preparation, train/test split, benchmark comparison, forecast evaluation, output creation, and a lightweight dashboard.

## What it includes

- **Real retail data:** UCI Sales Transactions Weekly—purchased quantities for 800 products across 52 weeks.
- **Models:** seasonal-naive benchmark and an interpretable linear-trend regression.
- **Evaluation:** MAE, RMSE, and MAPE on a 12-week holdout period.
- **Outputs:** CSV forecast, model scorecard, SVG charts, and a standalone HTML dashboard.

## Run it

```bash
curl --fail --location --output /tmp/uci_weekly_sales.zip 'https://archive.ics.uci.edu/static/public/396/sales+transactions+dataset+weekly.zip'
mkdir -p data/raw
unzip -p /tmp/uci_weekly_sales.zip Sales_Transactions_Dataset_Weekly.csv > data/raw/Sales_Transactions_Dataset_Weekly.csv
python3 scripts/prepare_uci_weekly_sales.py
python3 run_pipeline.py
python3 -m unittest discover -s tests -v
```

Open `outputs/dashboard.html` in a browser after running the pipeline.

## Project structure

```text
data/                 Processed real UCI demand data
outputs/              Forecast, metrics, charts, and dashboard
src/forecasting.py    Forecasting and evaluation utilities
tests/                Standard-library unit tests
run_pipeline.py       End-to-end pipeline entry point
```

## Business interpretation

The selected model is chosen by lowest holdout MAE. The source series is a real, public dataset, but it contains **relative week indices rather than calendar dates**; the project therefore does not infer holiday or promotion effects. A production system should add dated sales history, price, stock availability, holidays, and confirmed promotion calendars.

## Data source and attribution

Tan, S. C., & Lau, J. P. S. (2014). *Sales Transactions Weekly* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XS4Q. Licensed under CC BY 4.0.
