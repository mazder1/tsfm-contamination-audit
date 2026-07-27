"""Gap segmentation - the policy fixed before any model was scored.

The rule: split at every gap, drop segments too short to forecast. There is no
gap-length threshold, deliberately. These tests exist so the executed policy
cannot drift from the pre-registered one.
"""

import numpy as np
import pandas as pd
import pytest

from tsfm_audit import config
from tsfm_audit.series import Series, split_at_gaps


def _series(values, freq="h"):
    return Series(
        series_id="test:series",
        source="test",
        domain="test",
        freq=freq,
        timestamps=pd.date_range("2025-01-01", periods=len(values), freq=freq, tz="UTC"),
        values=np.array(values, dtype=float),
    )


def test_no_gaps_returns_one_segment():
    segs = split_at_gaps(_series([1.0] * 10), min_length=3)
    assert len(segs) == 1
    assert len(segs[0]) == 10
    assert segs[0].series_id == "test:series#s1"


def test_missing_value_splits():
    segs = split_at_gaps(_series([1, 2, 3, np.nan, 5, 6, 7]), min_length=3)
    assert [len(s) for s in segs] == [3, 3]
    assert [s.series_id for s in segs] == ["test:series#s1", "test:series#s2"]


def test_short_segments_dropped_but_numbering_stays_stable():
    # Segments of length 4, 1, 4 - the middle one cannot be forecast.
    segs = split_at_gaps(_series([1, 2, 3, 4, np.nan, 9, np.nan, 5, 6, 7, 8]), min_length=3)
    assert [len(s) for s in segs] == [4, 4]
    # The survivor keeps number 3, not 2. Renumbering would hide the drop.
    assert [s.series_id for s in segs] == ["test:series#s1", "test:series#s3"]
    assert segs[-1].metadata["segments_found"] == 3


def test_every_gap_splits_regardless_of_length():
    # A one-step gap splits exactly as a hundred-step gap would. This is the
    # property that removes the free parameter.
    one = _series([1, 2, 3, 4, np.nan, 5, 6, 7, 8])
    many = _series([1, 2, 3, 4] + [np.nan] * 20 + [5, 6, 7, 8])
    assert len(split_at_gaps(one, min_length=4)) == 2
    assert len(split_at_gaps(many, min_length=4)) == 2


def test_break_in_the_timestamp_grid_also_splits():
    # Values all present, but an hour is absent from the index entirely.
    stamps = pd.DatetimeIndex(
        list(pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"))
        + list(pd.date_range("2025-01-01 06:00", periods=4, freq="h", tz="UTC"))
    )
    s = Series(
        series_id="test:series",
        source="test",
        domain="test",
        freq="h",
        timestamps=stamps,
        values=np.arange(8, dtype=float),
    )
    assert [len(x) for x in split_at_gaps(s, min_length=3)] == [4, 4]


def test_all_missing_returns_nothing():
    assert split_at_gaps(_series([np.nan] * 5), min_length=1) == []


def test_segments_carry_parent_provenance():
    seg = split_at_gaps(_series([1, 2, 3, np.nan, 5, 6, 7]), min_length=3)[0]
    assert seg.metadata["parent_series_id"] == "test:series"
    assert seg.metadata["segment_number"] == 1


def test_min_usable_length_is_context_plus_horizon():
    assert config.min_usable_segment_length(512) == 512 + config.EVAL_HORIZON


def test_max_audited_context_is_unset_until_phase_1():
    # Mirrors Protocol.detection_floor: the rule is pre-registered, the number
    # is measured later. A value appearing here before Phase 1 means someone
    # guessed it, which is what the None is guarding against.
    assert config.MAX_AUDITED_CONTEXT is None
    with pytest.raises(ValueError, match="Phase 1"):
        config.min_usable_segment_length()
