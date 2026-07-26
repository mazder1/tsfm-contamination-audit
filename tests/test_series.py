"""Round-tripping and provenance on the shared series container."""

import numpy as np
import pandas as pd
import pytest

from tsfm_audit.series import Series, load_series, save_series, series_summary


def _series(series_id: str = "a") -> Series:
    return Series(
        series_id=series_id,
        source="test",
        domain="synthetic",
        freq="D",
        timestamps=pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC"),
        values=np.arange(8, dtype=float),
    )


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        Series(
            series_id="bad",
            source="test",
            domain="synthetic",
            freq="D",
            timestamps=pd.date_range("2025-01-01", periods=3, freq="D"),
            values=[1.0, 2.0],
        )


def test_content_hash_tracks_values_not_identity():
    assert _series("a").content_hash == _series("b").content_hash
    other = _series()
    other.values[0] = 99.0
    assert other.content_hash != _series().content_hash


def test_parquet_roundtrip_preserves_values(tmp_path):
    original = [_series("a"), _series("b")]
    path = save_series(original, tmp_path / "snap.parquet")
    restored = {s.series_id: s for s in load_series(path)}

    assert set(restored) == {"a", "b"}
    np.testing.assert_array_equal(restored["a"].values, original[0].values)
    assert restored["a"].content_hash == original[0].content_hash


def test_summary_reports_missing_values():
    s = _series()
    s.values[2] = np.nan
    assert series_summary([s])[0]["n_missing"] == 1
