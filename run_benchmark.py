"""Forecast the last 28 days of M5 (store x department level) and save the scores and the forecasts.

Usage: python run_benchmark.py
"""
from pathlib import Path

import pandas as pd

from forecasting.baselines import seasonal_naive_with_empirical_quantiles
from forecasting.data import load_store_department_daily_sales
from forecasting.metrics import interval_coverage, mean_absolute_scaled_error, scaled_quantile_loss

HORIZON_DAYS = 28  # the M5 competition horizon
QUANTILE_LEVELS = [0.1, 0.5, 0.9]  # 0.1 and 0.9 bound the 80% prediction interval
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
MODEL_NAME = "seasonal_naive"


def main():
    daily_sales = load_store_department_daily_sales()
    test_dates = daily_sales.columns[-HORIZON_DAYS:]
    score_rows, forecast_rows = [], []

    for series_id, sales in daily_sales.iterrows():
        training_history = sales.to_numpy(dtype=float)[:-HORIZON_DAYS]
        actual = sales.to_numpy(dtype=float)[-HORIZON_DAYS:]
        point_forecast, quantile_forecasts = seasonal_naive_with_empirical_quantiles(
            training_history, HORIZON_DAYS, QUANTILE_LEVELS)
        lower_bound_80, upper_bound_80 = quantile_forecasts[0], quantile_forecasts[-1]

        score_rows.append({
            "model": MODEL_NAME,
            "series_id": series_id,
            "mase": mean_absolute_scaled_error(actual, point_forecast, training_history),
            "scaled_quantile_loss": scaled_quantile_loss(actual, quantile_forecasts, QUANTILE_LEVELS, training_history),
            "coverage_80": interval_coverage(actual, lower_bound_80, upper_bound_80),
        })
        for day_index, date in enumerate(test_dates):
            forecast_rows.append({
                "model": MODEL_NAME,
                "series_id": series_id,
                "date": date.date().isoformat(),
                "actual": actual[day_index],
                "point_forecast": point_forecast[day_index],
                **{f"quantile_{level}": quantile_forecasts[level_index, day_index]
                   for level_index, level in enumerate(QUANTILE_LEVELS)},
            })

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(RESULTS_DIRECTORY / f"scores_{MODEL_NAME}.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(RESULTS_DIRECTORY / f"forecasts_{MODEL_NAME}.csv", index=False)
    metric_columns = ["mase", "scaled_quantile_loss", "coverage_80"]
    print(f"{len(scores)} series, test window {test_dates[0].date()} to {test_dates[-1].date()}")
    print(scores[metric_columns].mean().round(3).to_string())
    # Series ids look like "CA_1_FOODS_3": the category is the third part.
    scores["category"] = scores["series_id"].str.split("_").str[2]
    print("\nBy category:")
    print(scores.groupby("category")[metric_columns].mean().round(3).to_string())


if __name__ == "__main__":
    main()
