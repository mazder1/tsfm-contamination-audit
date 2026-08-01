"""Prove the surrogates preserve what they claim to preserve.

The Phase 3 gate. Each family asserts something specific, and an assertion that
is not measured is exactly what this project criticises elsewhere:

* **IAAFT** claims the marginal distribution *exactly* - the surrogate is a
  permutation of the original values - and the power spectrum approximately.
* **Block bootstrap** claims local dependence up to the block length, and
  explicitly does *not* claim the long-range structure it is designed to destroy.

This module measures those claims. It does not test whether the preserved
properties are *sufficient for forecasting* - that is a different and harder
question, and it is Phase 3.5's job. A surrogate can pass everything here and
still have removed something a model legitimately relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation for lags 1..max_lag."""
    values = np.asarray(series, dtype=float)
    values = values - values.mean()
    denom = float(np.dot(values, values))
    if denom == 0:
        return np.zeros(max_lag)
    return np.array(
        [float(np.dot(values[:-lag], values[lag:]) / denom) for lag in range(1, max_lag + 1)]
    )


def power_spectrum(series: np.ndarray) -> np.ndarray:
    """Power at each rfft frequency."""
    return np.abs(np.fft.rfft(np.asarray(series, dtype=float) - np.mean(series))) ** 2


def quantile_profile(series: np.ndarray, n_points: int = 101) -> np.ndarray:
    """The marginal distribution, as evenly spaced quantiles."""
    return np.quantile(np.asarray(series, dtype=float), np.linspace(0.0, 1.0, n_points))


@dataclass
class ValidationReport:
    family: str
    n_surrogates: int
    # Marginal distribution: max absolute difference between the real and
    # surrogate quantile profiles, relative to the real series' spread.
    distribution_max_rel_diff: float
    # Autocorrelation: max absolute difference over the compared lags. ACF is
    # already dimensionless, so this is an absolute difference, not relative.
    acf_max_abs_diff: float
    # Power spectrum: median relative difference across frequencies. Median, not
    # max, because individual high frequencies carry little power and produce
    # huge relative errors on almost no absolute error.
    spectrum_median_rel_diff: float
    acf_by_lag: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def validate(
    series: np.ndarray,
    surrogates: np.ndarray,
    family: str,
    max_lag: int = 50,
) -> ValidationReport:
    """Compare an ensemble of surrogates against the series they came from.

    Surrogate statistics are averaged across the ensemble before comparison: a
    single surrogate is a random draw and will differ from the original by
    chance, whereas the ensemble mean is what the family actually promises.
    """
    values = np.asarray(series, dtype=float)
    ensemble = np.atleast_2d(np.asarray(surrogates, dtype=float))
    max_lag = int(min(max_lag, len(values) // 4))

    spread = float(values.max() - values.min())
    spread = spread if spread > 0 else 1.0

    real_quantiles = quantile_profile(values)
    surrogate_quantiles = np.mean([quantile_profile(s) for s in ensemble], axis=0)
    distribution_diff = float(np.max(np.abs(real_quantiles - surrogate_quantiles)) / spread)

    real_acf = autocorrelation(values, max_lag)
    surrogate_acf = np.mean([autocorrelation(s, max_lag) for s in ensemble], axis=0)
    acf_diff = np.abs(real_acf - surrogate_acf)

    real_spectrum = power_spectrum(values)
    surrogate_spectrum = np.mean([power_spectrum(s) for s in ensemble], axis=0)
    scale = np.maximum(real_spectrum, real_spectrum.max() * 1e-12)
    spectrum_diff = float(np.median(np.abs(real_spectrum - surrogate_spectrum) / scale))

    return ValidationReport(
        family=family,
        n_surrogates=len(ensemble),
        distribution_max_rel_diff=distribution_diff,
        acf_max_abs_diff=float(np.max(acf_diff)) if max_lag else 0.0,
        spectrum_median_rel_diff=spectrum_diff,
        acf_by_lag=acf_diff,
    )
