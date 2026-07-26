"""ENTSO-E Transparency Platform — actual total electricity load.

Requires a free API token in ``ENTSOE_API_TOKEN``. Without one this source
reports itself unavailable and the fetch continues with the other sources; a
missing token is a configuration gap, not a pipeline failure.

Electricity load is the archetypal forecasting benchmark domain (ETT, the
Electricity dataset), which is precisely why a *fresh* electricity series is
worth having.
"""

from __future__ import annotations

import datetime as dt
import os
import xml.etree.ElementTree as ET

import pandas as pd

from ...series import Series
from ._http import get_text

SOURCE_KEY = "entsoe"
DOMAIN = "electricity"
FREQ = "h"
TOKEN_ENV = "ENTSOE_API_TOKEN"
API_URL = "https://web-api.tp.entsoe.eu/api"

# Bidding-zone EIC codes.
ZONES: tuple[tuple[str, str], ...] = (
    ("PL", "10YPL-AREA-----S"),
    ("DE_LU", "10Y1001A1001A82H"),
    ("ES", "10YES-REE------0"),
)

_RESOLUTION_MINUTES = {"PT15M": 15, "PT30M": 30, "PT60M": 60, "PT1H": 60}


def available() -> bool:
    return bool(os.environ.get(TOKEN_ENV))


def fetch(start: dt.date, end: dt.date) -> list[Series]:
    """Fetch hourly actual total load per bidding zone."""
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV} is not set. Request a free token at "
            "https://transparency.entsoe.eu/ (Account Settings) and export it."
        )

    out: list[Series] = []
    for zone_name, eic in ZONES:
        frames: list[pd.Series] = []
        # The API caps a request at one year; walk the window in yearly chunks.
        for chunk_start, chunk_end in _yearly_chunks(start, end):
            xml = get_text(
                API_URL,
                {
                    "securityToken": token,
                    "documentType": "A65",  # system total load
                    "processType": "A16",  # realised
                    "outBiddingZone_Domain": eic,
                    "periodStart": chunk_start.strftime("%Y%m%d%H%M"),
                    "periodEnd": chunk_end.strftime("%Y%m%d%H%M"),
                },
            )
            parsed = _parse_load_xml(xml)
            if parsed is not None and not parsed.empty:
                frames.append(parsed)

        if not frames:
            continue

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        # Sub-hourly resolutions are averaged up so every zone shares one grid.
        hourly = combined.resample("1h").mean()

        out.append(
            Series(
                series_id=f"{SOURCE_KEY}:load:{zone_name}",
                source=SOURCE_KEY,
                domain=DOMAIN,
                freq=FREQ,
                timestamps=pd.DatetimeIndex(hourly.index),
                values=hourly.to_numpy(dtype=float),
                metadata={"zone": zone_name, "eic": eic, "units": "MW", "document_type": "A65"},
            )
        )
    return out


def _yearly_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.datetime, dt.datetime]]:
    chunks: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = dt.datetime.combine(start, dt.time.min)
    stop = dt.datetime.combine(end, dt.time.min)
    while cursor < stop:
        nxt = min(cursor + dt.timedelta(days=365), stop)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_load_xml(xml: str) -> pd.Series | None:
    """Extract a timestamp-indexed load series from a GL_MarketDocument."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None

    index: list[pd.Timestamp] = []
    values: list[float] = []

    for period in root.iter():
        if _strip_ns(period.tag) != "Period":
            continue

        start_text = None
        resolution = None
        for child in period:
            name = _strip_ns(child.tag)
            if name == "timeInterval":
                for sub in child:
                    if _strip_ns(sub.tag) == "start":
                        start_text = sub.text
            elif name == "resolution":
                resolution = child.text

        if not start_text or resolution not in _RESOLUTION_MINUTES:
            continue

        step = dt.timedelta(minutes=_RESOLUTION_MINUTES[resolution])
        period_start = pd.Timestamp(start_text).tz_convert("UTC")

        for point in period:
            if _strip_ns(point.tag) != "Point":
                continue
            position = quantity = None
            for field in point:
                name = _strip_ns(field.tag)
                if name == "position":
                    position = int(field.text)
                elif name == "quantity":
                    quantity = float(field.text)
            if position is None or quantity is None:
                continue
            index.append(period_start + step * (position - 1))
            values.append(quantity)

    if not index:
        return None
    return pd.Series(values, index=pd.DatetimeIndex(index))
