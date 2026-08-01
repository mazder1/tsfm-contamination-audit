"""Surrogate families and their validation suite.

Each family asserts something specific about what it preserves. These tests hold
them to it - an unmeasured claim about the surrogates is exactly the failure this
project exists to detect in other people's work.
"""

import numpy as np
import pytest

from tsfm_audit.surrogates.block_bootstrap import (
    block_bootstrap_ensemble,
    moving_block_bootstrap,
    suggest_block_length,
)
from tsfm_audit.surrogates.iaaft import iaaft, iaaft_ensemble
from tsfm_audit.surrogates.validation import autocorrelation, validate


def seasonal_series(n: int = 512, period: int = 24) -> np.ndarray:
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(0)
    return 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)


# --- IAAFT ---------------------------------------------------------------


def test_iaaft_preserves_the_value_distribution_exactly():
    # The strong claim: the surrogate is a permutation of the original values.
    # The rank-ordering step runs last precisely so this holds exactly.
    series = seasonal_series()
    result = iaaft(series, seed=1)
    np.testing.assert_allclose(np.sort(result.surrogate), np.sort(series))


def test_iaaft_approximately_preserves_the_power_spectrum():
    series = seasonal_series()
    surrogates = iaaft_ensemble(series, seeds=[1, 2, 3, 4, 5])
    report = validate(series, surrogates, family="iaaft")
    assert report.spectrum_median_rel_diff < 0.25


def test_iaaft_preserves_autocorrelation():
    series = seasonal_series()
    surrogates = iaaft_ensemble(series, seeds=list(range(8)))
    report = validate(series, surrogates, family="iaaft")
    assert report.acf_max_abs_diff < 0.2


def test_iaaft_destroys_instance_identity():
    # The whole point: same statistics, different series. If a surrogate were
    # nearly identical to the original, a gap would prove nothing.
    series = seasonal_series()
    surrogate = iaaft(series, seed=1).surrogate
    assert np.max(np.abs(surrogate - series)) > 0.1 * (series.max() - series.min())


def test_iaaft_is_reproducible_from_its_seed():
    # Surrogates are never stored, only their seeds, so identical seeds must
    # give identical surrogates or nothing is reproducible.
    series = seasonal_series()
    np.testing.assert_array_equal(
        iaaft(series, seed=42).surrogate, iaaft(series, seed=42).surrogate
    )
    assert not np.array_equal(iaaft(series, seed=1).surrogate, iaaft(series, seed=2).surrogate)


def test_iaaft_rejects_unusable_input():
    with pytest.raises(ValueError, match="missing values"):
        iaaft(np.array([1.0, np.nan] * 8), seed=0)
    with pytest.raises(ValueError, match="too short"):
        iaaft(np.arange(4, dtype=float), seed=0)


# --- Block bootstrap -----------------------------------------------------


def test_block_bootstrap_returns_original_length_from_original_values():
    series = seasonal_series()
    surrogate = moving_block_bootstrap(series, seed=1, block_length=24)
    assert len(surrogate) == len(series)
    assert set(np.unique(surrogate)).issubset(set(np.unique(series)))


def test_block_bootstrap_preserves_short_range_dependence():
    # Its actual claim: structure *inside* a block survives. Compare against a
    # full shuffle, which preserves the distribution but no dependence at all.
    series = seasonal_series()
    rng = np.random.default_rng(0)

    blocked = block_bootstrap_ensemble(series, seeds=list(range(8)), block_length=48)
    shuffled = np.stack([rng.permutation(series) for _ in range(8)])

    real = autocorrelation(series, 6)
    blocked_acf = np.mean([autocorrelation(s, 6) for s in blocked], axis=0)
    shuffled_acf = np.mean([autocorrelation(s, 6) for s in shuffled], axis=0)

    assert np.mean(np.abs(blocked_acf - real)) < np.mean(np.abs(shuffled_acf - real))


def test_block_bootstrap_destroys_long_range_structure():
    # It is *supposed* to. Preserving long-range order would leave the instance
    # recognisable, which would defeat the probe.
    series = seasonal_series(n=1024)
    surrogates = block_bootstrap_ensemble(series, seeds=list(range(8)), block_length=24)
    report = validate(series, surrogates, family="block_bootstrap", max_lag=200)
    assert report.acf_max_abs_diff > 0.1


def test_block_bootstrap_is_reproducible_from_its_seed():
    series = seasonal_series()
    np.testing.assert_array_equal(
        moving_block_bootstrap(series, seed=7, block_length=24),
        moving_block_bootstrap(series, seed=7, block_length=24),
    )


def test_suggested_block_length_holds_a_whole_season():
    series = seasonal_series(n=1000)
    # n**(1/3) is 10 here, so the seasonal period must win.
    assert suggest_block_length(series, season_length=24) == 24
    # With no seasonality the asymptotic rate stands alone.
    assert suggest_block_length(series, season_length=1) == 10


# --- Validation suite ----------------------------------------------------


def test_validation_flags_a_surrogate_that_preserves_nothing():
    # A surrogate of pure noise shares neither spectrum nor distribution, and
    # the suite must say so - otherwise it would pass anything.
    series = seasonal_series()
    rng = np.random.default_rng(0)
    fake = rng.normal(0, 1, (5, len(series)))
    report = validate(series, fake, family="broken")
    assert report.distribution_max_rel_diff > 0.1
