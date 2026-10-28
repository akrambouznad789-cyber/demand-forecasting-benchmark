"""Check the seasonal naive baseline on series whose correct forecast is known in advance."""
import numpy as np
import pytest

from forecasting.baselines import seasonal_naive_with_empirical_quantiles

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
WEEKLY_PATTERN = np.array([10, 12, 14, 16, 30, 40, 20], dtype=float)


def test_a_perfectly_weekly_series_is_forecast_exactly_with_a_zero_width_interval():
    training_history = np.tile(WEEKLY_PATTERN, 60)
    point_forecast, quantile_forecasts = seasonal_naive_with_empirical_quantiles(training_history, 28, QUANTILE_LEVELS)
    np.testing.assert_allclose(point_forecast, np.tile(WEEKLY_PATTERN, 4))
    for quantile_forecast in quantile_forecasts:
        np.testing.assert_allclose(quantile_forecast, point_forecast)


def test_every_forecast_week_repeats_the_last_observed_week():
    random_generator = np.random.default_rng(seed=0)
    training_history = random_generator.poisson(lam=50, size=420).astype(float)
    point_forecast, _ = seasonal_naive_with_empirical_quantiles(training_history, 28, QUANTILE_LEVELS)
    np.testing.assert_allclose(point_forecast, np.tile(training_history[-7:], 4))


def test_quantiles_are_ordered_and_never_negative():
    random_generator = np.random.default_rng(seed=1)
    # Mostly zero-sale days, like intermittent item-level demand.
    training_history = random_generator.poisson(lam=0.3, size=420).astype(float)
    _, quantile_forecasts = seasonal_naive_with_empirical_quantiles(training_history, 28, QUANTILE_LEVELS)
    assert (quantile_forecasts >= 0).all()
    assert (np.diff(quantile_forecasts, axis=0) >= 0).all()


def test_the_interval_widens_with_the_horizon_on_a_random_walk():
    # On a random walk the error of "repeat the value from k weeks ago" grows with k, so the 80% interval must too.
    random_generator = np.random.default_rng(seed=2)
    training_history = 1000 + np.cumsum(random_generator.normal(size=2000))
    _, quantile_forecasts = seasonal_naive_with_empirical_quantiles(training_history, 28, QUANTILE_LEVELS)
    interval_width = quantile_forecasts[-1] - quantile_forecasts[0]
    mean_width_per_week = interval_width.reshape(4, 7).mean(axis=1)
    assert mean_width_per_week[3] > mean_width_per_week[0]


@pytest.mark.parametrize("horizon_days", [1, 7, 28])
def test_output_shapes_match_the_horizon(horizon_days):
    training_history = np.tile(WEEKLY_PATTERN, 60)
    point_forecast, quantile_forecasts = seasonal_naive_with_empirical_quantiles(
        training_history, horizon_days, QUANTILE_LEVELS)
    assert point_forecast.shape == (horizon_days,)
    assert quantile_forecasts.shape == (len(QUANTILE_LEVELS), horizon_days)
