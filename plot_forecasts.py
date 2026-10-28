"""Draw the README figure: the last 8 weeks of history and the 28-day forecast with its 80% interval, for two
series picked by rule (not by eye): the one with the median MASE and the one with the lowest interval coverage.

Usage: python plot_forecasts.py   (run run_benchmark.py first)
Writes figures/forecast_examples_light.png and figures/forecast_examples_dark.png (GitHub shows the one matching
the reader's theme).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from forecasting.data import load_store_department_daily_sales

PROJECT_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = PROJECT_DIRECTORY / "results"
FIGURES_DIRECTORY = PROJECT_DIRECTORY / "figures"
MODEL_NAME = "seasonal_naive"
HISTORY_DAYS_SHOWN = 56

# Colors validated for color-vision deficiency and contrast against each surface.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary_ink": "#52514e", "muted_ink": "#898781",
              "gridline": "#e1e0d9", "baseline": "#c3c2b7", "actual": "#2a78d6", "forecast": "#eb6834",
              "interval_opacity": 0.18},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "secondary_ink": "#c3c2b7", "muted_ink": "#898781",
             "gridline": "#2c2c2a", "baseline": "#383835", "actual": "#3987e5", "forecast": "#d95926",
             "interval_opacity": 0.30},
}


def pick_example_series(scores: pd.DataFrame) -> list[tuple[str, str]]:
    """(series_id, why it was picked) for the median-MASE series and the lowest-coverage series."""
    median_mase = scores["mase"].median()
    typical_series = scores.loc[(scores["mase"] - median_mase).abs().idxmin(), "series_id"]
    lowest_coverage_series = scores.loc[scores["coverage_80"].idxmin(), "series_id"]
    return [(typical_series, "typical series (median MASE)"), (lowest_coverage_series, "lowest interval coverage")]


def draw_figure(daily_sales, forecasts, scores, theme_name: str) -> Path:
    colors = THEMES[theme_name]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.6), facecolor=colors["surface"])

    for axis, (series_id, reason) in zip(axes, pick_example_series(scores)):
        series_forecast = forecasts[forecasts["series_id"] == series_id]
        forecast_dates = pd.to_datetime(series_forecast["date"])
        sales = daily_sales.loc[series_id]
        # One continuous line: the last weeks of training history, then the actual demand of the test window.
        actual_demand_shown = sales[sales.index <= forecast_dates.max()].iloc[-(HISTORY_DAYS_SHOWN + len(forecast_dates)):]
        series_scores = scores[scores["series_id"] == series_id].iloc[0]

        axis.set_facecolor(colors["surface"])
        axis.plot(actual_demand_shown.index, actual_demand_shown.to_numpy(), color=colors["actual"], linewidth=1.6,
                  label="Actual demand")
        axis.plot(forecast_dates, series_forecast["point_forecast"], color=colors["forecast"], linewidth=1.6,
                  linestyle=(0, (4, 2)), label="Forecast (seasonal naive)")
        axis.fill_between(forecast_dates, series_forecast["quantile_0.1"], series_forecast["quantile_0.9"],
                          color=colors["forecast"], alpha=colors["interval_opacity"], linewidth=0,
                          label="80% prediction interval")
        forecast_origin = forecast_dates.min() - pd.Timedelta(hours=12)
        axis.axvline(forecast_origin, color=colors["muted_ink"], linewidth=0.8)
        axis.annotate("Forecast origin", xy=(forecast_origin, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(-5, -2), textcoords="offset points", ha="right", va="top", color=colors["muted_ink"],
                      fontsize=9)

        axis.set_title(f"{series_id}  ·  {reason}  ·  MASE {series_scores['mase']:.2f}  ·  "
                       f"80% interval coverage {series_scores['coverage_80']:.2f}",
                       loc="left", color=colors["ink"], fontsize=11, pad=8)
        axis.set_ylabel("Units sold per day", color=colors["secondary_ink"])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color=colors["gridline"], linewidth=0.8)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.tick_params(colors=colors["muted_ink"], length=0, pad=6)
        for side in ["top", "right", "left"]:
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(colors["baseline"])

    legend = figure.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False,
                           bbox_to_anchor=(0.06, 1.0))
    for text in legend.get_texts():
        text.set_color(colors["secondary_ink"])
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)

    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    output_path = FIGURES_DIRECTORY / f"forecast_examples_{theme_name}.png"
    figure.savefig(output_path, dpi=160, facecolor=colors["surface"])
    plt.close(figure)
    return output_path


def main():
    daily_sales = load_store_department_daily_sales()
    forecasts = pd.read_csv(RESULTS_DIRECTORY / f"forecasts_{MODEL_NAME}.csv")
    scores = pd.read_csv(RESULTS_DIRECTORY / f"scores_{MODEL_NAME}.csv")
    for theme_name in THEMES:
        print(f"Wrote {draw_figure(daily_sales, forecasts, scores, theme_name)}")


if __name__ == "__main__":
    main()
