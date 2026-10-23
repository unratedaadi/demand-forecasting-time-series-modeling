import unittest

from datetime import date, timedelta

from src.forecasting import (
    evaluate_forecast,
    make_time_features,
    recursive_linear_forecast,
    aggregate_weekly_sales,
)


class TimeFeatureTests(unittest.TestCase):
    def test_time_features_include_lags_and_calendar_columns(self):
        frame = [
            {"date": date(2025, 1, 1) + timedelta(days=index), "demand": float(10 + index)}
            for index in range(10)
        ]

        featured = make_time_features(frame, lags=(1, 7), rolling_window=3)

        self.assertIn("lag_1", featured[0])
        self.assertIn("lag_7", featured[0])
        self.assertIn("rolling_mean_3", featured[0])
        self.assertIn("day_of_week", featured[0])
        self.assertEqual(featured[0]["date"], date(2025, 1, 8))
        self.assertEqual(featured[0]["lag_1"], 16.0)

    def test_aggregates_real_retail_product_rows_by_week(self):
        rows = [
            {"Product_Code": "P1", "W0": "11", "W1": "12"},
            {"Product_Code": "P2", "W0": "7", "W1": "6"},
        ]
        self.assertEqual(aggregate_weekly_sales(rows), [{"week": 0, "demand": 18.0, "source": "UCI Sales Transactions Weekly"}, {"week": 1, "demand": 18.0, "source": "UCI Sales Transactions Weekly"}])


class ForecastTests(unittest.TestCase):
    def test_recursive_forecast_returns_requested_future_dates(self):
        history = [
            {"date": date(2025, 1, 1) + timedelta(days=index), "demand": float(100 + index)}
            for index in range(45)
        ]

        forecast = recursive_linear_forecast(history, horizon=7)

        self.assertEqual(len(forecast), 7)
        self.assertEqual(forecast[0]["date"], date(2025, 2, 15))
        self.assertTrue(all(row["forecast"] >= 0 for row in forecast))

    def test_evaluation_reports_standard_regression_metrics(self):
        result = evaluate_forecast([10.0, 20.0], [12.0, 18.0])

        self.assertAlmostEqual(result["mae"], 2.0)
        self.assertAlmostEqual(result["rmse"], 2.0)
        self.assertAlmostEqual(result["mape"], 15.0)


if __name__ == "__main__":
    unittest.main()
