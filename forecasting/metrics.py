"""Accuracy and calibration metrics for point and probabilistic forecasts.

Errors are scaled by the in-sample error of a seasonal naive forecast, so series with very different
volumes (a small hobby department vs. a large food department) can be averaged together.
"""
import numpy as np


def in_sample_seasonal_naive_error(training_history: np.ndarray, season_length: int = 7) -> float:
    return float(np.mean(np.abs(training_history[season_length:] - training_history[:-season_length])))


def mean_absolute_scaled_error(actual, point_forecast, training_history, season_length: int = 7) -> float:
    scale = in_sample_seasonal_naive_error(training_history, season_length)
    return float(np.mean(np.abs(actual - point_forecast)) / scale)


def scaled_quantile_loss(actual, quantile_forecasts, quantile_levels, training_history, season_length: int = 7) -> float:
    """Pinball loss averaged over the quantile levels, scaled like MASE.
    With many evenly spaced levels, twice this value approximates the scaled CRPS."""
    scale = in_sample_seasonal_naive_error(training_history, season_length)
    pinball_losses = []
    for quantile_level, quantile_forecast in zip(quantile_levels, quantile_forecasts):
        error = actual - quantile_forecast
        pinball_losses.append(np.mean(np.maximum(quantile_level * error, (quantile_level - 1) * error)))
    return float(np.mean(pinball_losses) / scale)


def interval_coverage(actual, lower_bound, upper_bound) -> float:
    """Share of actual values inside the prediction interval; an 80% interval should cover about 0.80."""
    return float(np.mean((actual >= lower_bound) & (actual <= upper_bound)))
