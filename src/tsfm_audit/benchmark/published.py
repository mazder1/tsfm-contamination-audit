"""The published Chronos zero-shot benchmark: configs, loading, and splitting.

Phase 1 reproduces ``chronos-t5-base-zero-shot.csv`` and
``seasonal-naive-zero-shot.csv`` from the Chronos repository. The backtest
configs below are transcribed from that repository's
``scripts/evaluation/configs/zero-shot.yaml`` and vendored here so the
reproduction is pinned against a moving upstream file.

The splitting logic mirrors the reference: a series is cut at ``offset`` and the
next ``prediction_length`` points are the target. Every config uses one window
per series, so there is no rolling evaluation to reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHRONOS_REPO = "amazon-science/chronos-forecasting"
RESULTS_URL = (
    "https://raw.githubusercontent.com/amazon-science/chronos-forecasting/"
    "main/scripts/evaluation/results/{model}-zero-shot.csv"
)


@dataclass(frozen=True)
class BacktestConfig:
    name: str
    hf_repo: str
    offset: int
    prediction_length: int
    num_rolls: int = 1


_MAIN = "autogluon/chronos_datasets"
_EXTRA = "autogluon/chronos_datasets_extra"

# ETT cannot be redistributed under its licence, so the reference dataset builds
# it on the fly from a loader script. ``datasets`` 5.x removed script-based
# loading entirely, so we replicate the builder's ETT generator directly against
# the same upstream CSVs. Transcribed from chronos_datasets_extra.py:
# two regions per name, each contributing every non-timestamp column as its own
# univariate series.
_ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}{region}.csv"
_ETT_NAMES = ("ETTh", "ETTm")

# Transcribed from zero-shot.yaml. Order preserved from the source file.
ZERO_SHOT: tuple[BacktestConfig, ...] = (
    BacktestConfig("monash_traffic", _MAIN, -24, 24),
    BacktestConfig("monash_australian_electricity", _MAIN, -48, 48),
    BacktestConfig("ercot", _MAIN, -24, 24),
    BacktestConfig("ETTm", _EXTRA, -96, 24),
    BacktestConfig("ETTh", _EXTRA, -24, 24),
    BacktestConfig("exchange_rate", _MAIN, -30, 30),
    BacktestConfig("nn5", _MAIN, -56, 56),
    BacktestConfig("monash_nn5_weekly", _MAIN, -8, 8),
    BacktestConfig("monash_weather", _MAIN, -30, 30),
    BacktestConfig("monash_covid_deaths", _MAIN, -30, 30),
    BacktestConfig("monash_fred_md", _MAIN, -12, 12),
    BacktestConfig("m4_quarterly", _MAIN, -8, 8),
    BacktestConfig("m4_yearly", _MAIN, -6, 6),
    BacktestConfig("dominick", _MAIN, -8, 8),
    BacktestConfig("m5", _MAIN, -28, 28),
    BacktestConfig("monash_tourism_monthly", _MAIN, -24, 24),
    BacktestConfig("monash_tourism_quarterly", _MAIN, -8, 8),
    BacktestConfig("monash_tourism_yearly", _MAIN, -4, 4),
    BacktestConfig("monash_car_parts", _MAIN, -12, 12),
    BacktestConfig("monash_hospital", _MAIN, -12, 12),
    BacktestConfig("monash_cif_2016", _MAIN, -12, 12),
    BacktestConfig("monash_m1_yearly", _MAIN, -6, 6),
    BacktestConfig("monash_m1_quarterly", _MAIN, -8, 8),
    BacktestConfig("monash_m1_monthly", _MAIN, -18, 18),
    BacktestConfig("monash_m3_monthly", _MAIN, -18, 18),
    BacktestConfig("monash_m3_yearly", _MAIN, -6, 6),
    BacktestConfig("monash_m3_quarterly", _MAIN, -8, 8),
)

BY_NAME = {c.name: c for c in ZERO_SHOT}

# The four datasets that dominate total series count, split out so the sweep can
# be staged: everything else first, these as a separate long run.
LARGE = ("dominick", "m5", "m4_quarterly", "m4_yearly")


@dataclass
class BacktestWindow:
    """One evaluation instance: history, the target that follows it, and freq."""

    past: np.ndarray
    target: np.ndarray
    freq: str


def split_series(values: np.ndarray, offset: int, prediction_length: int) -> BacktestWindow | None:
    """Cut a series at ``offset``; the next ``prediction_length`` points are the target.

    ``offset`` is negative, counted from the end. Returns None when the series is
    too short to yield both a history and a full target.

    The end index needs care: with ``offset=-12`` and ``prediction_length=12``
    the naive ``values[offset : offset + prediction_length]`` becomes
    ``values[-12:0]``, which is empty rather than the last twelve points. Most
    configs in this benchmark have exactly that shape.
    """
    values = np.asarray(values, dtype=float)
    past = values[:offset]
    end = offset + prediction_length
    target = (values[offset:] if end >= 0 else values[offset:end])[:prediction_length]
    if len(past) == 0 or len(target) < prediction_length:
        return None
    return BacktestWindow(past=past, target=target, freq="")


def _load_ett_frames(name: str) -> list[pd.DataFrame]:
    """Fetch the raw ETT CSVs the reference loader script would have built from."""
    frames = []
    for region in (1, 2):
        frame = pd.read_csv(_ETT_URL.format(name=name, region=region), parse_dates=["date"])
        frames.append(frame.rename(columns={"date": "timestamp"}))
    return frames


def _ett_windows(config: BacktestConfig) -> list[BacktestWindow]:
    windows: list[BacktestWindow] = []
    freq = ""
    for frame in _load_ett_frames(config.name):
        if not freq:
            freq = pd.DatetimeIndex(frame["timestamp"]).to_period()[0].freqstr
        for column in frame.columns:
            if column == "timestamp":
                continue
            window = split_series(frame[column].to_numpy(), config.offset, config.prediction_length)
            if window is not None:
                window.freq = freq
                windows.append(window)
    return windows


def load_windows(config: BacktestConfig) -> list[BacktestWindow]:
    """Load one benchmark dataset and produce its evaluation windows.

    Mirrors the reference loader: every sequence column other than ``timestamp``
    becomes an independent univariate series, and the frequency is taken from
    the first row on the assumption - the reference's, not ours - that a dataset
    has one frequency throughout.
    """
    if config.name in _ETT_NAMES:
        return _ett_windows(config)

    import datasets

    ds = datasets.load_dataset(config.hf_repo, config.name, split="train")
    ds.set_format("numpy")

    sequence_fields = [
        name
        for name, feature in ds.features.items()
        if isinstance(feature, datasets.Sequence) and name != "timestamp"
    ]

    freq = pd.DatetimeIndex(ds[0]["timestamp"]).to_period()[0].freqstr

    windows: list[BacktestWindow] = []
    for row in ds:
        for field in sequence_fields:
            window = split_series(row[field], config.offset, config.prediction_length)
            if window is not None:
                window.freq = freq
                windows.append(window)
    return windows


def load_published_results(model: str = "chronos-t5-base") -> pd.DataFrame:
    """Fetch the reference result CSV we are trying to reproduce."""
    return pd.read_csv(RESULTS_URL.format(model=model)).set_index("dataset")
