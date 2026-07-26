"""Open-Meteo historical weather archive.

Free, no API key, deep archive with a few days' lag. Hourly 2m temperature at a
spread of stations chosen for climate diversity — a monoculture benchmark would
only test one kind of seasonality.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ...series import Series
from ._http import get_json

SOURCE_KEY = "open_meteo"
DOMAIN = "weather"
FREQ = "h"
API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLE = "temperature_2m"

# (id, latitude, longitude) — deliberately spread across climate zones and
# hemispheres so seasonality is not identical across series.
LOCATIONS: tuple[tuple[str, float, float], ...] = (
    ("reykjavik", 64.1466, -21.9426),
    ("warsaw", 52.2297, 21.0122),
    ("phoenix", 33.4484, -112.0740),
    ("singapore", 1.3521, 103.8198),
    ("nairobi", -1.2921, 36.8219),
    ("sydney", -33.8688, 151.2093),
)


def fetch(start: dt.date, end: dt.date) -> list[Series]:
    """Fetch hourly temperature for every configured location."""
    out: list[Series] = []
    for name, lat, lon in LOCATIONS:
        payload = get_json(
            API_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "hourly": VARIABLE,
                "timezone": "UTC",
            },
        )
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        values = hourly.get(VARIABLE) or []
        if not times:
            continue
        out.append(
            Series(
                series_id=f"{SOURCE_KEY}:{VARIABLE}:{name}",
                source=SOURCE_KEY,
                domain=DOMAIN,
                freq=FREQ,
                timestamps=pd.DatetimeIndex(pd.to_datetime(times, utc=True)),
                values=np.array([np.nan if v is None else float(v) for v in values], dtype=float),
                metadata={
                    "latitude": lat,
                    "longitude": lon,
                    "variable": VARIABLE,
                    "units": (payload.get("hourly_units") or {}).get(VARIABLE),
                    "api": API_URL,
                },
            )
        )
    return out
