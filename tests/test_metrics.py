"""Check each metric against values computed by hand."""
import numpy as np
import pytest

from forecasting.metrics import (in_sample_seasonal_naive_error, interval_coverage, mean_absolute_scaled_error,
                                 scaled_quantile_loss)

# Two weeks whose second week is the first week plus one unit: the in-sample seasonal naive error is exactly 1.
TRAINING_HISTORY_WITH_UNIT_SCALE = np.array([1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8], dtype=float)


def test_in_sample_seasonal_naive_error_is_the_mean_weekly_change():
    assert in_sample_seasonal_naive_error(TRAINING_HISTORY_WITH_UNIT_SCALE) == pytest.approx(1.0)


def test_mase_is_mean_absolute_error_divided_by_the_scale():
    actual = np.array([10.0, 10.0])
    point_forecast = np.array([8.0, 11.0])
    assert mean_absolute_scaled_error(actual, point_forecast, TRAINING_HISTORY_WITH_UNIT_SCALE) == pytest.approx(1.5)


def test_mase_is_zero_for_a_perfect_forecast():
    actual = np.array([3.0, 5.0, 7.0])
    assert mean_absolute_scaled_error(actual, actual, TRAINING_HISTORY_WITH_UNIT_SCALE) == 0.0


def test_median_quantile_loss_is_half_the_mase():
    actual = np.array([10.0, 4.0, 6.0])
    median_forecast = np.array([7.0, 5.0, 6.0])
    quantile_loss = scaled_quantile_loss(actual, [median_forecast], [0.5], TRAINING_HISTORY_WITH_UNIT_SCALE)
    mase = mean_absolute_scaled_error(actual, median_forecast, TRAINING_HISTORY_WITH_UNIT_SCALE)
    assert quantile_loss == pytest.approx(0.5 * mase)


def test_high_quantile_penalizes_under_forecasting_more_than_over_forecasting():
    # At the 0.9 quantile, being 10 units too low costs 0.9 x 10 = 9; being 10 units too high costs 0.1 x 10 = 1.
    actual = np.array([100.0])
    under_forecast_loss = scaled_quantile_loss(actual, [np.array([90.0])], [0.9], TRAINING_HISTORY_WITH_UNIT_SCALE)
    over_forecast_loss = scaled_quantile_loss(actual, [np.array([110.0])], [0.9], TRAINING_HISTORY_WITH_UNIT_SCALE)
    assert under_forecast_loss == pytest.approx(9.0)
    assert over_forecast_loss == pytest.approx(1.0)


def test_interval_coverage_counts_values_on_the_bounds_as_inside():
    actual = np.array([1.0, 5.0, 9.0, 12.0])
    lower_bound = np.array([1.0, 0.0, 0.0, 0.0])
    upper_bound = np.array([10.0, 10.0, 9.0, 10.0])
    assert interval_coverage(actual, lower_bound, upper_bound) == pytest.approx(0.75)
