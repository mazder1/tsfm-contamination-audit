"""Guards on the single function the fresh benchmark's guarantee rests on."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tsfm_audit import config
from tsfm_audit.benchmark.fetch import enforce_admissibility
from tsfm_audit.series import Series

CUTOFF = dt.date(2025, 1, 1)


def _series(start: str, periods: int, tz: str | None = "UTC") -> Series:
    return Series(
        series_id="test:series",
        source="test",
        domain="synthetic",
        freq="D",
        timestamps=pd.date_range(start, periods=periods, freq="D", tz=tz),
        values=np.arange(periods, dtype=float),
    )


def test_drops_observations_before_cutoff():
    kept, dropped = enforce_admissibility([_series("2024-12-25", 14)], CUTOFF)
    assert dropped == 7
    assert len(kept) == 1
    assert kept[0].timestamps.min() >= pd.Timestamp(CUTOFF, tz="UTC")


def test_keeps_fully_admissible_series_untouched():
    kept, dropped = enforce_admissibility([_series("2025-06-01", 10)], CUTOFF)
    assert dropped == 0
    assert len(kept[0]) == 10


def test_drops_series_that_is_entirely_pre_cutoff():
    kept, dropped = enforce_admissibility([_series("2020-01-01", 5)], CUTOFF)
    assert kept == []
    assert dropped == 5


def test_naive_timestamps_are_treated_as_utc_not_rejected():
    kept, _ = enforce_admissibility([_series("2025-06-01", 5, tz=None)], CUTOFF)
    assert kept[0].timestamps.tz is not None


def test_values_stay_aligned_with_timestamps_after_trimming():
    kept, _ = enforce_admissibility([_series("2024-12-30", 6)], CUTOFF)
    # Original values are 0..5 starting 2024-12-30; the first two are dropped.
    assert kept[0].values.tolist() == [2.0, 3.0, 4.0, 5.0]


def test_run_refuses_a_start_before_the_pre_registered_cutoff():
    from tsfm_audit.benchmark import fetch

    earlier = config.FRESH_BENCHMARK_START - dt.timedelta(days=1)
    with pytest.raises(ValueError, match="precedes the pre-registered cutoff"):
        fetch.run(start=earlier)
