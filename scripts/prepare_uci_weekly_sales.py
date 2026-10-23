"""Prepare UCI's real Sales Transactions Weekly data for the forecast pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting import aggregate_weekly_sales

RAW_PATH = ROOT / "data" / "raw" / "Sales_Transactions_Dataset_Weekly.csv"
OUTPUT_PATH = ROOT / "data" / "uci_weekly_sales_demand.csv"


def main() -> None:
    with RAW_PATH.open(newline="", encoding="utf-8") as file:
        rows = aggregate_weekly_sales(csv.DictReader(file))
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["week", "demand", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} real weekly observations at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
