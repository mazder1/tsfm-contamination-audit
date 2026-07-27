"""Seasonal naive: MASE's own denominator, and the first baseline that matters.

Deliberately the first forecaster implemented. It has no training corpus, so
memorization is impossible by construction - which makes it both the Phase 2
baseline and one of the Phase 3.5 null-control forecasters.
"""

from __future__ import annotations

import numpy as np


def seasonal_naive_forecast(past: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    """Repeat the last seasonal cycle forward.

    Falls back to the last observation (lag 1) when the history is shorter than
    one season, matching the seasonal-error fallback in :mod:`analysis.metrics`.
    """
    past = np.asarray(past, dtype=float)
    if len(past) == 0:
        raise ValueError("cannot forecast from an empty history")
    lag = season_length if len(past) >= season_length else 1
    cycle = past[-lag:]
    reps = int(np.ceil(horizon / lag))
    return np.tile(cycle, reps)[:horizon]


def seasonal_naive_quantiles(
    past: np.ndarray,
    horizon: int,
    season_length: int,
    quantile_levels: list[float],
) -> np.ndarray:
    """Point forecast broadcast across quantile levels, shape ``(horizon, n_levels)``.

    Seasonal naive is deterministic, so every quantile is the same value. The
    quantile loss of a point forecast is still well defined, which is how the
    reference implementation scores it.
    """
    point = seasonal_naive_forecast(past, horizon, season_length)
    return np.repeat(point[:, None], len(quantile_levels), axis=1)
