"""Dependency-free utilities for a small, interpretable demand forecast."""

from __future__ import annotations

import math
from datetime import date, timedelta
from statistics import mean
from typing import Dict, Iterable, List, Sequence


Row = Dict[str, object]

def aggregate_weekly_sales(rows: Iterable[Dict[str, str]]) -> List[Row]:
    """Sum the 800 product-level UCI series into real weekly total demand."""
    rows = list(rows)
    totals = [0.0] * 52
    for row in rows:
        for week in range(52):
            if f"W{week}" in row:
                totals[week] += float(row[f"W{week}"])
    observed_weeks = max((int(key[1:]) for row in rows for key in row if key.startswith("W") and key[1:].isdigit()), default=-1) + 1
    return [{"week": week, "demand": total, "source": "UCI Sales Transactions Weekly"} for week, total in enumerate(totals[:observed_weeks])]


def make_time_features(rows: Sequence[Row], lags: Sequence[int] = (1, 7), rolling_window: int = 7) -> List[Row]:
    """Add prior-demand and calendar features without using future observations."""
    highest_lag = max(max(lags), rolling_window)
    featured: List[Row] = []
    for index in range(highest_lag, len(rows)):
        current = dict(rows[index])
        current_date = current["date"]
        if not isinstance(current_date, date):
            raise TypeError("Each row's date must be a datetime.date")
        for lag in lags:
            current[f"lag_{lag}"] = float(rows[index - lag]["demand"])
        window_values = [float(rows[position]["demand"]) for position in range(index - rolling_window, index)]
        current[f"rolling_mean_{rolling_window}"] = mean(window_values)
        current["day_of_week"] = current_date.weekday()
        current["month"] = current_date.month
        featured.append(current)
    return featured


def seasonal_naive_forecast(history: Sequence[Row], horizon: int, season_length: int = 7, frequency_days: int = 1) -> List[Row]:
    """Repeat the most recent weekly demand pattern into the future."""
    if len(history) < season_length:
        raise ValueError("History must contain at least one full seasonal cycle")
    latest_date = history[-1]["date"]
    if not isinstance(latest_date, date):
        raise TypeError("Each row's date must be a datetime.date")
    pattern = [float(row["demand"]) for row in history[-season_length:]]
    return [
        {"date": latest_date + timedelta(days=step * frequency_days), "forecast": max(0.0, pattern[(step - 1) % season_length])}
        for step in range(1, horizon + 1)
    ]


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """Solve an augmented linear system with Gauss-Jordan elimination."""
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-10:
            raise ValueError("Feature matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * reference for value, reference in zip(augmented[row], augmented[column])]
    return [augmented[row][-1] for row in range(size)]


def _design_row(current_date: date, index: int) -> List[float]:
    """Return an intercept and index trend for datasets without calendar dates."""
    return [1.0, float(index)]


def fit_linear_model(history: Sequence[Row]) -> List[float]:
    """Fit a ridge-regularized least-squares linear trend model."""
    features = [_design_row(row["date"], index) for index, row in enumerate(history)]
    targets = [float(row["demand"]) for row in history]
    width = len(features[0])
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for feature_row, target in zip(features, targets):
        for left in range(width):
            xty[left] += feature_row[left] * target
            for right in range(width):
                xtx[left][right] += feature_row[left] * feature_row[right]
    for index in range(width):
        xtx[index][index] += 0.01
    return _solve_linear_system(xtx, xty)


def recursive_linear_forecast(history: Sequence[Row], horizon: int, frequency_days: int = 1) -> List[Row]:
    """Predict a future horizon with the fitted linear-trend model."""
    if not history:
        raise ValueError("History cannot be empty")
    coefficients = fit_linear_model(history)
    latest_date = history[-1]["date"]
    if not isinstance(latest_date, date):
        raise TypeError("Each row's date must be a datetime.date")
    forecast: List[Row] = []
    for step in range(1, horizon + 1):
        future_date = latest_date + timedelta(days=step * frequency_days)
        features = _design_row(future_date, len(history) + step - 1)
        prediction = max(0.0, sum(coefficient * value for coefficient, value in zip(coefficients, features)))
        forecast.append({"date": future_date, "forecast": round(prediction, 2)})
    return forecast


def evaluate_forecast(actual: Iterable[float], predicted: Iterable[float]) -> Dict[str, float]:
    """Return MAE, RMSE, and percentage error while safely handling zeros."""
    actual_values = [float(value) for value in actual]
    predicted_values = [float(value) for value in predicted]
    if not actual_values or len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted values must be non-empty and equal in length")
    errors = [observed - estimated for observed, estimated in zip(actual_values, predicted_values)]
    percentage_errors = [abs(error / observed) * 100 for observed, error in zip(actual_values, errors) if observed != 0]
    return {
        "mae": round(mean(abs(error) for error in errors), 4),
        "rmse": round(math.sqrt(mean(error * error for error in errors)), 4),
        "mape": round(mean(percentage_errors), 4) if percentage_errors else 0.0,
    }
