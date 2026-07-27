"""Scoring rules.

The seasonality values here are not guesses - each was confirmed by reproducing
the published seasonal-naive MASE on a dataset of that frequency to within
0.03 percent. Locking them in a test stops a silent change to the mapping from
quietly rescaling every MASE in the project.
"""

import numpy as np
import pytest

from tsfm_audit.analysis.metrics import (
    get_seasonality,
    mase,
    quantile_loss,
    seasonal_error,
    weighted_quantile_loss,
)
from tsfm_audit.baselines.seasonal_naive import seasonal_naive_forecast


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("h", 24),
        ("H", 24),
        ("15min", 96),
        ("30min", 48),
        ("D", 1),
        ("W-SUN", 1),
        ("W-THU", 1),
        ("M", 12),
        ("Q-DEC", 4),
        ("Q-OCT", 4),
        ("Y-DEC", 1),
        ("B", 5),
    ],
)
def test_seasonality_matches_the_reproduced_benchmark(freq, expected):
    assert get_seasonality(freq) == expected


def test_unknown_frequency_is_non_seasonal():
    assert get_seasonality("7min") == 1
    assert get_seasonality("") == 1


def test_seasonal_error_is_mean_absolute_seasonal_difference():
    past = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert seasonal_error(past, 1) == pytest.approx(1.0)
    assert seasonal_error(past, 2) == pytest.approx(2.0)


def test_seasonal_error_falls_back_to_lag_one_when_history_is_short():
    assert seasonal_error(np.array([1.0, 2.0, 3.0]), 24) == pytest.approx(1.0)


def test_mase_of_a_perfect_forecast_is_zero():
    past = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    target = np.array([7.0, 8.0])
    assert mase(target, target, past, 1) == pytest.approx(0.0)


def test_seasonal_naive_reproduces_a_perfect_cycle():
    cycle = np.array([1.0, 5.0, 3.0, 9.0])
    past = np.tile(cycle, 10)
    forecast = seasonal_naive_forecast(past, 8, 4)
    np.testing.assert_allclose(forecast, np.tile(cycle, 2))


def test_mase_is_undefined_when_the_seasonal_error_is_zero():
    # A perfectly periodic series has zero seasonal error, so MASE has a zero
    # denominator and is genuinely undefined - not zero. Returning nan here is
    # what keeps such series out of the aggregate instead of silently scoring
    # them as perfect.
    past = np.tile(np.array([1.0, 5.0, 3.0, 9.0]), 10)
    target = np.array([1.0, 5.0, 3.0, 9.0])
    forecast = seasonal_naive_forecast(past, 4, 4)
    assert np.isnan(mase(target, forecast, past, 4))


def test_mase_of_seasonal_naive_on_a_cycle_with_drift():
    # Same cycle plus a slow drift, so the seasonal error is non-zero and the
    # forecast is good but not exact.
    cycle = np.array([1.0, 5.0, 3.0, 9.0])
    past = np.tile(cycle, 10) + 0.1 * np.arange(40)
    target = np.tile(cycle, 1) + 0.1 * np.arange(40, 44)
    forecast = seasonal_naive_forecast(past, 4, 4)
    score = mase(target, forecast, past, 4)
    assert np.isfinite(score)
    assert 0.0 < score < 5.0


def test_quantile_loss_is_asymmetric_around_the_level():
    target = np.array([10.0])
    # Under-forecasting at q=0.9 should cost more than over-forecasting.
    under = quantile_loss(target, np.array([8.0]), 0.9).sum()
    over = quantile_loss(target, np.array([12.0]), 0.9).sum()
    assert under > over
    # At the median the penalty is symmetric.
    assert quantile_loss(target, np.array([8.0]), 0.5).sum() == pytest.approx(
        quantile_loss(target, np.array([12.0]), 0.5).sum()
    )


def test_weighted_quantile_loss_of_a_perfect_forecast_is_zero():
    targets = [np.array([1.0, 2.0, 3.0])]
    forecasts = [np.repeat(targets[0][:, None], 3, axis=1)]
    assert weighted_quantile_loss(targets, forecasts, [0.1, 0.5, 0.9]) == pytest.approx(0.0)


def test_weighted_quantile_loss_pools_across_series():
    # A large-magnitude series must dominate a tiny one, because the metric
    # pools totals rather than averaging per-series ratios.
    big = np.array([1000.0, 1000.0])
    small = np.array([0.001, 0.001])
    exact_big = [np.repeat(big[:, None], 1, axis=1), np.repeat(small[:, None], 1, axis=1)]
    wrong_small = [
        np.repeat(big[:, None], 1, axis=1),
        np.repeat((small * 2)[:, None], 1, axis=1),
    ]
    assert weighted_quantile_loss([big, small], exact_big, [0.5]) == pytest.approx(0.0)
    # Doubling the tiny series' forecast barely moves the pooled metric.
    assert weighted_quantile_loss([big, small], wrong_small, [0.5]) < 1e-5
