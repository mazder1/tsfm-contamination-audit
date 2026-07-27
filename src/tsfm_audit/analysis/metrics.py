"""Scoring rules: MASE, weighted quantile loss, and the seasonality they depend on.

These are written to match the definitions used by the published Chronos
evaluation (gluonts ``MASE`` and ``MeanWeightedSumQuantileLoss``), because
Phase 1 reproduces those numbers. They are our own implementation on purpose:
running the reference implementation would validate the reference, not us.
"""

from __future__ import annotations

import re

import numpy as np

# Base seasonality per pandas offset alias, in periods of that alias.
# Mirrors gluonts' DEFAULT_SEASONALITIES.
_BASE_SEASONALITY: dict[str, int] = {
    "S": 3600,  # second -> hour
    "s": 3600,
    "T": 1440,  # minute -> day
    "min": 1440,
    "H": 24,  # hour -> day
    "h": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "ME": 12,
    "B": 5,
    "Q": 4,
    "QE": 4,
    "A": 1,
    "Y": 1,
    "YE": 1,
}

_FREQ_RE = re.compile(r"^(\d*)\s*([A-Za-z]+)")


def get_seasonality(freq: str) -> int:
    """Seasonal period for a pandas frequency string, e.g. ``H`` -> 24.

    A multiplied frequency divides the base period when it divides evenly
    (``15T`` -> 1440/15 = 96) and is otherwise treated as non-seasonal.
    """
    match = _FREQ_RE.match(freq.strip())
    if not match:
        return 1
    multiple_text, alias = match.groups()
    multiple = int(multiple_text) if multiple_text else 1

    base = _BASE_SEASONALITY.get(alias)
    if base is None:
        base = _BASE_SEASONALITY.get(alias[0])
    if base is None:
        return 1

    if multiple == 1:
        return base
    return base // multiple if base % multiple == 0 else 1


def seasonal_error(past: np.ndarray, season_length: int) -> float:
    """Mean absolute seasonal difference of the in-sample data.

    This is MASE's denominator. Falls back to a lag of 1 when the history is
    too short for the seasonal lag, matching the reference behaviour.
    """
    past = np.asarray(past, dtype=float)
    lag = season_length if len(past) >= season_length else 1
    if len(past) <= lag:
        return float("nan")
    diffs = np.abs(past[lag:] - past[:-lag])
    return float(np.nanmean(diffs))


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    past: np.ndarray,
    season_length: int,
) -> float:
    """MASE for one series: horizon MAE scaled by the in-sample seasonal error."""
    target = np.asarray(target, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = seasonal_error(past, season_length)
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float(np.nanmean(np.abs(target - forecast)) / denom)


def quantile_loss(target: np.ndarray, prediction: np.ndarray, q: float) -> np.ndarray:
    """Pinball loss, in the doubled convention used by the reference metric."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    under = target - prediction
    return 2.0 * np.where(under >= 0, q * under, (q - 1.0) * under)


def weighted_quantile_loss(
    targets: list[np.ndarray],
    quantile_forecasts: list[np.ndarray],
    quantile_levels: list[float],
) -> float:
    """Mean weighted sum quantile loss, pooled across series.

    For each level the total pinball loss over every series and timestep is
    divided by the total absolute target, then averaged across levels. Pooling
    rather than averaging per-series ratios is what the reference does, and it
    matters: per-series ratios would let tiny-magnitude series dominate.
    """
    denom = float(sum(np.nansum(np.abs(np.asarray(t, dtype=float))) for t in targets))
    if denom == 0:
        return float("nan")

    per_level = []
    for i, q in enumerate(quantile_levels):
        total = 0.0
        for target, forecast in zip(targets, quantile_forecasts, strict=True):
            total += float(np.nansum(quantile_loss(target, np.asarray(forecast)[:, i], q)))
        per_level.append(total / denom)
    return float(np.mean(per_level))
