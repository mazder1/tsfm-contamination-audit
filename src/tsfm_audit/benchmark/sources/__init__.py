"""Live data sources feeding the fresh benchmark.

Each module exposes ``fetch(start, end) -> list[Series]`` and a ``SOURCE_KEY``.
"""

from __future__ import annotations

from collections.abc import Callable

from . import entsoe, open_meteo, wikipedia

FETCHERS: dict[str, Callable] = {
    open_meteo.SOURCE_KEY: open_meteo.fetch,
    wikipedia.SOURCE_KEY: wikipedia.fetch,
    entsoe.SOURCE_KEY: entsoe.fetch,
}

__all__ = ["FETCHERS", "entsoe", "open_meteo", "wikipedia"]
