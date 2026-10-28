"""Draw the README figures from the benchmark results (run run_benchmark.py first).

Usage: python plot_forecasts.py
Each figure is written to figures/ in a light and a dark version; GitHub shows the one matching the reader's theme.
- forecast_explained: how to read a probabilistic forecast, on the typical series
- coverage_by_series: does the 80% range hold? One dot per store-department
- forecast_examples: the typical series and the series with the lowest interval coverage
Example series are picked by rule (median MASE, lowest coverage), not by eye.
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
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]

# Colors validated for color-vision deficiency and contrast against each surface.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary_ink": "#52514e", "muted_ink": "#898781",
              "gridline": "#e1e0d9", "baseline": "#c3c2b7", "actual": "#2a78d6", "forecast": "#eb6834",
              "interval_opacity": 0.18},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "secondary_ink": "#c3c2b7", "muted_ink": "#898781",
             "gridline": "#2c2c2a", "baseline": "#383835", "actual": "#3987e5", "forecast": "#d95926",
             "interval_opacity": 0.22},
}


def pick_example_series(scores: pd.DataFrame) -> list[tuple[str, str]]:
    """(series_id, why it was picked) for the median-MASE series and the lowest-coverage series."""
    median_mase = scores["mase"].median()
    typical_series = scores.loc[(scores["mase"] - median_mase).abs().idxmin(), "series_id"]
    lowest_coverage_series = scores.loc[scores["coverage_80"].idxmin(), "series_id"]
    return [(typical_series, "typical series (median MASE)"), (lowest_coverage_series, "lowest interval coverage")]


def style_axis(axis, colors):
    axis.set_facecolor(colors["surface"])
    axis.set_axisbelow(True)
    axis.tick_params(colors=colors["muted_ink"], length=0, pad=6)
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(colors["baseline"])


def draw_forecast_with_interval(axis, sales, series_forecast, colors):
    """Actual demand as one continuous line (history, then test window), the forecast and its 80% interval."""
    forecast_dates = pd.to_datetime(series_forecast["date"])
    actual_demand_shown = sales[sales.index <= forecast_dates.max()].iloc[-(HISTORY_DAYS_SHOWN + len(forecast_dates)):]
    axis.plot(actual_demand_shown.index, actual_demand_shown.to_numpy(), color=colors["actual"], linewidth=1.6,
              label="Actual demand")
    axis.plot(forecast_dates, series_forecast["point_forecast"], color=colors["forecast"], linewidth=1.6,
              linestyle=(0, (4, 2)), label="Forecast (seasonal naive)")
    axis.fill_between(forecast_dates, series_forecast["quantile_0.1"], series_forecast["quantile_0.9"],
                      color=colors["forecast"], alpha=colors["interval_opacity"], linewidth=0,
                      label="80% range (prediction interval)")
    forecast_origin = forecast_dates.min() - pd.Timedelta(hours=12)
    axis.axvline(forecast_origin, color=colors["muted_ink"], linewidth=0.8)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("Units sold per day", color=colors["secondary_ink"])
    axis.grid(axis="y", color=colors["gridline"], linewidth=0.8)
    style_axis(axis, colors)
    return forecast_origin


def draw_forecast_explained(daily_sales, forecasts, scores, colors):
    series_id = pick_example_series(scores)[0][0]
    series_forecast = forecasts[forecasts["series_id"] == series_id].reset_index(drop=True)
    forecast_dates = pd.to_datetime(series_forecast["date"])
    figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=colors["surface"])
    forecast_origin = draw_forecast_with_interval(axis, daily_sales.loc[series_id], series_forecast, colors)

    outside = (series_forecast["actual"] < series_forecast["quantile_0.1"]) | (
        series_forecast["actual"] > series_forecast["quantile_0.9"])
    axis.scatter(forecast_dates[outside], series_forecast.loc[outside, "actual"], s=46, color=colors["surface"],
                 edgecolors=colors["actual"], linewidths=1.8, zorder=4, label="Day outside the range")
    axis.set_ylim(0, series_forecast["actual"].max() * 1.32)

    annotation_style = {"color": colors["ink"], "fontsize": 9.5,
                        "arrowprops": {"arrowstyle": "-", "color": colors["secondary_ink"], "linewidth": 0.8}}
    axis.annotate("Past sales: what the method sees", xy=(forecast_origin, 1.0), xycoords=("data", "axes fraction"),
                  xytext=(-8, -4), textcoords="offset points", ha="right", va="top", color=colors["muted_ink"])
    axis.annotate("Next 4 weeks: what it must predict", xy=(forecast_origin, 1.0), xycoords=("data", "axes fraction"),
                  xytext=(8, -4), textcoords="offset points", ha="left", va="top", color=colors["muted_ink"])

    widest_day = (series_forecast["quantile_0.9"] - series_forecast["quantile_0.1"]).idxmax()
    axis.annotate("80% range: the method expects\n8 days out of 10 to land here",
                  xy=(forecast_dates[widest_day], series_forecast.loc[widest_day, "quantile_0.1"] + 20),
                  xytext=(forecast_dates[widest_day] + pd.Timedelta(days=2), 60), ha="left", **annotation_style)
    second_forecast_day = 1
    axis.annotate("Forecast: the method's\nbest single guess",
                  xy=(forecast_dates[second_forecast_day], series_forecast.loc[second_forecast_day, "point_forecast"]),
                  xytext=(forecast_origin - pd.Timedelta(days=1), 200), ha="right", **annotation_style)
    highest_surprise_day = (series_forecast["actual"] - series_forecast["quantile_0.9"]).idxmax()
    axis.annotate("Above the range: more demand than\nplanned, risk of empty shelves",
                  xy=(forecast_dates[highest_surprise_day], series_forecast.loc[highest_surprise_day, "actual"]),
                  xytext=(forecast_origin + pd.Timedelta(days=1), series_forecast["actual"].max() * 1.1), ha="left",
                  **annotation_style)

    axis.set_xlim(forecast_origin - pd.Timedelta(days=21), forecast_dates.max() + pd.Timedelta(days=1))
    axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.set_title(f"How to read a probabilistic forecast  ·  {series_id}, Walmart store-department, 2016",
                   loc="left", color=colors["ink"], fontsize=11, pad=30)
    legend = axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(colors["secondary_ink"])
    figure.tight_layout()
    return figure


def draw_coverage_by_series(scores, colors):
    scores = scores.assign(category=scores["series_id"].str.split("_").str[2],
                           coverage_percent=scores["coverage_80"] * 100)
    figure, axis = plt.subplots(figsize=(10, 3.9), facecolor=colors["surface"])
    for row_index, category in enumerate(CATEGORIES):
        category_scores = scores[scores["category"] == category].sort_values("coverage_percent")
        # Coverage is a count of days out of 28, so several series share a value: stack them so every dot shows.
        stack_position = category_scores.groupby("coverage_percent").cumcount()
        stack_size = category_scores.groupby("coverage_percent")["coverage_percent"].transform("size")
        row_positions = row_index + (stack_position - (stack_size - 1) / 2) * 0.13
        axis.scatter(category_scores["coverage_percent"], row_positions, s=42, color=colors["actual"],
                     edgecolors=colors["surface"], linewidths=1.2, zorder=3)

    axis.axvline(80, color=colors["secondary_ink"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    axis.annotate("Target: 80% of days", xy=(80, -0.75), xytext=(5, 0), textcoords="offset points", va="center",
                  color=colors["ink"], fontsize=9.5)
    axis.annotate("← Too narrow: more surprises\n    than the method announced", xy=(31, -0.75), va="center",
                  color=colors["muted_ink"], fontsize=9)
    axis.annotate("Too wide: more safety →\nstock than needed    ", xy=(102.5, -0.75), ha="right", va="center",
                  color=colors["muted_ink"], fontsize=9)
    lowest = scores.loc[scores["coverage_percent"].idxmin()]
    axis.annotate(f"{lowest['series_id']}: {lowest['coverage_percent']:.0f}%\n(demand jumped, see Figure 3)",
                  xy=(lowest["coverage_percent"], CATEGORIES.index(lowest["category"])), xytext=(10, -16),
                  textcoords="offset points", va="top", color=colors["ink"], fontsize=9,
                  arrowprops={"arrowstyle": "-", "color": colors["secondary_ink"], "linewidth": 0.8})

    category_counts = scores["category"].value_counts()
    axis.set_yticks(range(len(CATEGORIES)), [f"{category}\n{category_counts[category]} series" for category in CATEGORIES],
                    color=colors["secondary_ink"])
    axis.set_ylim(len(CATEGORIES) - 0.4, -1.1)
    axis.set_xlim(30, 103)
    axis.set_xticks(range(40, 101, 10), [f"{value}%" for value in range(40, 101, 10)])
    axis.set_xlabel("Share of the 28 test days that fell inside the 80% range", color=colors["secondary_ink"])
    axis.grid(axis="x", color=colors["gridline"], linewidth=0.8)
    style_axis(axis, colors)
    axis.set_title("Does the 80% range hold?  ·  One dot per store-department, seasonal naive baseline",
                   loc="left", color=colors["ink"], fontsize=11, pad=10)
    figure.tight_layout()
    return figure


def draw_forecast_examples(daily_sales, forecasts, scores, colors):
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.6), facecolor=colors["surface"])
    for axis, (series_id, reason) in zip(axes, pick_example_series(scores)):
        series_forecast = forecasts[forecasts["series_id"] == series_id]
        forecast_origin = draw_forecast_with_interval(axis, daily_sales.loc[series_id], series_forecast, colors)
        axis.annotate("Forecast origin", xy=(forecast_origin, 1.0), xycoords=("data", "axes fraction"),
                      xytext=(-5, -2), textcoords="offset points", ha="right", va="top", color=colors["muted_ink"],
                      fontsize=9)
        series_scores = scores[scores["series_id"] == series_id].iloc[0]
        axis.set_title(f"{series_id}  ·  {reason}  ·  MASE {series_scores['mase']:.2f}  ·  "
                       f"80% interval coverage {series_scores['coverage_80']:.2f}",
                       loc="left", color=colors["ink"], fontsize=11, pad=8)
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    legend = figure.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=3, frameon=False,
                           bbox_to_anchor=(0.06, 1.0))
    for text in legend.get_texts():
        text.set_color(colors["secondary_ink"])
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    return figure


def main():
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    daily_sales = load_store_department_daily_sales()
    forecasts = pd.read_csv(RESULTS_DIRECTORY / f"forecasts_{MODEL_NAME}.csv")
    scores = pd.read_csv(RESULTS_DIRECTORY / f"scores_{MODEL_NAME}.csv")
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    for theme_name, colors in THEMES.items():
        figures = {
            "forecast_explained": draw_forecast_explained(daily_sales, forecasts, scores, colors),
            "coverage_by_series": draw_coverage_by_series(scores, colors),
            "forecast_examples": draw_forecast_examples(daily_sales, forecasts, scores, colors),
        }
        for figure_name, figure in figures.items():
            output_path = FIGURES_DIRECTORY / f"{figure_name}_{theme_name}.png"
            figure.savefig(output_path, dpi=160, facecolor=colors["surface"])
            plt.close(figure)
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
