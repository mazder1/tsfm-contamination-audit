"""Live smoke tests. Network-marked; excluded from CI.

Run locally with:  uv run pytest -m network
"""

import datetime as dt

import pytest

from tsfm_audit.benchmark.sources import entsoe, open_meteo, wikipedia

START = dt.date(2025, 1, 1)
END = dt.date(2025, 1, 31)

pytestmark = pytest.mark.network


def test_open_meteo_returns_hourly_series():
    series = open_meteo.fetch(START, END)
    assert series
    assert len(series[0]) > 24 * 25


def test_wikipedia_returns_daily_series():
    series = wikipedia.fetch(START, END)
    assert series
    assert 25 <= len(series[0]) <= 32


def test_entsoe_requires_a_token():
    if entsoe.available():
        pytest.skip("token present; covered by the fetch run")
    with pytest.raises(RuntimeError, match=entsoe.TOKEN_ENV):
        entsoe.fetch(START, END)
