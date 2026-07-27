"""Splitting the published benchmark series.

The end-index case matters: with offset -12 and prediction_length 12, a naive
slice becomes values[-12:0], which is empty. Most configs in the Chronos
zero-shot benchmark have exactly that shape, so this silently produced zero
evaluation windows until it was caught.
"""

import numpy as np

from tsfm_audit.benchmark import published


def test_offset_plus_horizon_is_zero_takes_the_tail():
    values = np.arange(100, dtype=float)
    w = published.split_series(values, offset=-12, prediction_length=12)
    assert w is not None
    assert len(w.past) == 88
    assert len(w.target) == 12
    np.testing.assert_array_equal(w.target, np.arange(88, 100, dtype=float))


def test_offset_deeper_than_horizon():
    # ETTm's shape: cut 96 back, then forecast only the next 24.
    values = np.arange(500, dtype=float)
    w = published.split_series(values, offset=-96, prediction_length=24)
    assert w is not None
    assert len(w.past) == 404
    np.testing.assert_array_equal(w.target, np.arange(404, 428, dtype=float))


def test_series_too_short_is_dropped():
    assert published.split_series(np.arange(12, dtype=float), -12, 12) is None
    assert published.split_series(np.arange(5, dtype=float), -12, 12) is None


def test_every_zero_shot_config_yields_a_window_on_a_long_series():
    values = np.arange(2000, dtype=float)
    for config in published.ZERO_SHOT:
        w = published.split_series(values, config.offset, config.prediction_length)
        assert w is not None, config.name
        assert len(w.target) == config.prediction_length, config.name
