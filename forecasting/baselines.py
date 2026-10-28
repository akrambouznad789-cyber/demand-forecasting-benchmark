"""Baseline forecasters that the foundation models must beat."""
import numpy as np


def seasonal_naive_with_empirical_quantiles(training_history: np.ndarray, horizon_days: int, quantile_levels,
                                            season_length: int = 7, residual_window_days: int = 364):
    """Repeat the last observed week. The prediction interval comes from the errors this same rule made
    over the past year, measured separately for 1, 2, 3 and 4 weeks ahead, because errors grow with the horizon.

    Returns (point_forecast, quantile_forecasts) with shapes (horizon,) and (len(quantile_levels), horizon).
    """
    last_index = len(training_history) - 1
    point_forecast = np.empty(horizon_days)
    quantile_forecasts = np.empty((len(quantile_levels), horizon_days))
    for step in range(1, horizon_days + 1):
        weeks_ahead = int(np.ceil(step / season_length))
        lag_days = weeks_ahead * season_length
        point = training_history[last_index + step - lag_days]
        past_errors_at_this_lag = training_history[-residual_window_days:] - training_history[-residual_window_days - lag_days:-lag_days]
        point_forecast[step - 1] = point
        # Unit sales cannot be negative.
        quantile_forecasts[:, step - 1] = np.maximum(0.0, point + np.quantile(past_errors_at_this_lag, quantile_levels))
    return point_forecast, quantile_forecasts
