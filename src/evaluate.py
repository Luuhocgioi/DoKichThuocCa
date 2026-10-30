"""Calculate measurement errors from an evaluation CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def calculate(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
    if not np.any(valid):
        raise ValueError("Không có cặp số đo hợp lệ để tính MAE/MAPE.")
    error = np.abs(actual[valid] - predicted[valid])
    mae = float(np.mean(error))
    mape = float(np.mean(error / np.abs(actual[valid])) * 100.0)
    return mae, mape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Đường dẫn CSV đánh giá")
    args = parser.parse_args()
    csv_path = Path(args.csv)

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "actual_length_cm", "actual_width_cm",
        "predicted_length_cm", "predicted_width_cm",
    }
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"CSV cần các cột: {', '.join(sorted(required))}")

    for dimension in ("length", "width"):
        actual = np.array(
            [float(row[f"actual_{dimension}_cm"]) for row in rows]
        )
        predicted = np.array(
            [float(row[f"predicted_{dimension}_cm"]) for row in rows]
        )
        mae, mape = calculate(actual, predicted)
        print(f"{dimension.title()}: MAE={mae:.3f} cm, MAPE={mape:.2f}%")
    print(f"Số mẫu: {len(rows)}")


if __name__ == "__main__":
    main()

