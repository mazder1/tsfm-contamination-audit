"""Gap segmentation - the policy fixed before any model was scored.

The rule: split at every gap, drop segments too short to forecast. There is no
gap-length threshold, deliberately. These tests exist so the executed policy
cannot drift from the pre-registered one.
"""

import numpy as np
import pandas as pd

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
    assert config.min_usable_segment_length() == config.EVAL_CONTEXT + config.EVAL_HORIZON
    assert config.min_usable_segment_length() == 536


def test_eval_context_fits_every_capped_model():
    # The pre-registered context must be one every audited model can actually
    # accept. If a checkpoint caps below it, that model would silently truncate
    # and be scored on less history than the others.
    for model in config.AUDITED_MODELS:
        if model.context_cap is not None:
            assert config.EVAL_CONTEXT <= model.context_cap, model.key


def test_eval_context_is_the_largest_that_fits_everywhere():
    # 512 is not arbitrary: it is the tightest cap across the audit. Raising it
    # would truncate Chronos and TimesFM; lowering it would handicap them for
    # no reason.
    caps = [m.context_cap for m in config.AUDITED_MODELS if m.context_cap is not None]
    assert caps, "no model reports a context cap - the pinned configs were not read"
    assert config.EVAL_CONTEXT == min(caps)
