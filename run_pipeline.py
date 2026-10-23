"""Run the complete simulated retail demand forecasting workflow."""

from __future__ import annotations

import csv
from datetime import date
from html import escape
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from src.forecasting import (
    evaluate_forecast,
    generate_synthetic_demand,
    recursive_linear_forecast,
    seasonal_naive_forecast,
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def svg_line_chart(path: Path, title: str, series: List[Dict[str, object]], colors: List[str]) -> None:
    width, height, margin = 960, 380, 55
    all_values = [float(value) for item in series for value in item["values"]]
    low, high = min(all_values), max(all_values)
    value_range = max(high - low, 1.0)
    chart_width, chart_height = width - margin * 2, height - margin * 2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="{margin}" y="30" font-family="Arial" font-size="18" font-weight="bold">{escape(title)}</text>']
    for grid in range(5):
        value = low + (value_range * grid / 4)
        y = margin + chart_height - (value - low) / value_range * chart_height
        parts.append(f'<line x1="{margin}" x2="{width - margin}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="8" y="{y + 4:.1f}" font-family="Arial" font-size="11" fill="#4b5563">{value:.0f}</text>')
    for line, color in zip(series, colors):
        values = line["values"]
        count = max(len(values) - 1, 1)
        points = []
        for index, value in enumerate(values):
            x = margin + chart_width * index / count
            y = margin + chart_height - (float(value) - low) / value_range * chart_height
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
    legend_x = width - margin - 250
    for index, (line, color) in enumerate(zip(series, colors)):
        x = legend_x + index * 120
        parts.append(f'<line x1="{x}" x2="{x + 18}" y1="42" y2="42" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 24}" y="46" font-family="Arial" font-size="11">{escape(str(line["name"]))}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def build_dashboard(metrics: Dict[str, Dict[str, float]], selected_model: str, future_forecast: List[Dict[str, object]]) -> None:
    cards = "".join(
        f'<article><h3>{escape(model)}</h3><p>MAE <strong>{scores["mae"]:.2f}</strong></p><p>RMSE {scores["rmse"]:.2f} · MAPE {scores["mape"]:.2f}%</p></article>'
        for model, scores in metrics.items()
    )
    forecast_rows = "".join(f'<tr><td>{row["date"]}</td><td>{row["forecast"]:.2f}</td></tr>' for row in future_forecast[:10])
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Demand Forecasting Dashboard</title><style>body{{font-family:Arial,sans-serif;max-width:1000px;margin:36px auto;color:#172554;padding:0 18px}}.notice{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}article{{padding:16px;border:1px solid #dbeafe;border-radius:10px;background:#f8fbff}}img{{width:100%;border:1px solid #e5e7eb;border-radius:8px;margin:18px 0}}table{{border-collapse:collapse;width:360px}}td,th{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left}}</style></head><body><h1>Retail Demand Forecasting</h1><p class="notice">This dashboard uses simulated retail-demand data for portfolio demonstration. It is not a production business forecast.</p><h2>Model selection</h2><p>The selected model is <strong>{escape(selected_model)}</strong>, based on the lowest holdout MAE.</p><section class="grid">{cards}</section><h2>Demand history</h2><img src="demand_history.svg" alt="Recent demand history"><h2>Forecast comparison</h2><img src="forecast_comparison.svg" alt="Test-period forecast comparison"><h2>Next 10 days</h2><table><tr><th>Date</th><th>Forecast units</th></tr>{forecast_rows}</table></body></html>'''
    (OUTPUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")


def main() -> None:
    data = generate_synthetic_demand(date(2024, 1, 1), days=540, seed=42)
    write_csv(DATA_DIR / "synthetic_demand.csv", data, ["date", "demand", "promotion", "data_type"])
    train, test = data[:-90], data[-90:]
    baseline = seasonal_naive_forecast(train, len(test))
    linear = recursive_linear_forecast(train, len(test))
    actual = [float(row["demand"]) for row in test]
    metrics = {
        "Seasonal naive": evaluate_forecast(actual, [float(row["forecast"]) for row in baseline]),
        "Trend + calendar": evaluate_forecast(actual, [float(row["forecast"]) for row in linear]),
    }
    selected_model = min(metrics, key=lambda name: metrics[name]["mae"])
    future_forecast = seasonal_naive_forecast(data, 30) if selected_model == "Seasonal naive" else recursive_linear_forecast(data, 30)
    for row in future_forecast:
        row["model"] = selected_model
        row["data_type"] = "simulated"
    write_csv(OUTPUT_DIR / "forecast.csv", future_forecast, ["date", "forecast", "model", "data_type"])
    write_csv(OUTPUT_DIR / "model_metrics.csv", [{"model": name, **scores} for name, scores in metrics.items()], ["model", "mae", "rmse", "mape"])
    svg_line_chart(OUTPUT_DIR / "demand_history.svg", "Simulated demand — last 120 days", [{"name": "Actual demand", "values": [float(row["demand"]) for row in data[-120:]]}], ["#2563eb"])
    svg_line_chart(OUTPUT_DIR / "forecast_comparison.svg", "Holdout forecast comparison", [{"name": "Actual", "values": actual}, {"name": "Selected forecast", "values": [float(row["forecast"]) for row in baseline if selected_model == "Seasonal naive"] or [float(row["forecast"]) for row in linear]}], ["#1d4ed8", "#f97316"])
    build_dashboard(metrics, selected_model, future_forecast)
    print(f"Selected model: {selected_model}")
    print(f"30-day simulated forecast average: {mean(float(row['forecast']) for row in future_forecast):.2f} units")
    print(f"Dashboard: {OUTPUT_DIR / 'dashboard.html'}")


if __name__ == "__main__":
    main()
