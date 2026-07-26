"""Wikimedia pageviews — daily view counts per article.

Free, no key, well documented, and a completely different generating process
from weather or electricity: bursty, human-driven, heavy-tailed.

Note for the audit: TimesFM is documented as training on Wikipedia pageviews,
so this source is *deliberately* adversarial — the domain is in-distribution
for at least one audited model while the observations are not. That is exactly
the recall-vs-transfer distinction we want the fresh benchmark to expose.
"""

from __future__ import annotations

import datetime as dt
from urllib.parse import quote

import numpy as np
import pandas as pd

from ...series import Series
from ._http import get_json

SOURCE_KEY = "wikipedia"
DOMAIN = "web_traffic"
FREQ = "D"
API_TEMPLATE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "{project}/all-access/user/{article}/daily/{start}/{end}"
)
PROJECT = "en.wikipedia"

# Articles with steady, non-trivial traffic and different shapes: evergreen
# reference, seasonal, weekday-driven, and event-driven.
ARTICLES: tuple[str, ...] = (
    "Python_(programming_language)",
    "Association_football",
    "Climate_change",
    "Stock_market",
    "Influenza",
    "Christmas",
)


def fetch(start: dt.date, end: dt.date) -> list[Series]:
    """Fetch daily pageviews for every configured article."""
    out: list[Series] = []
    for article in ARTICLES:
        url = API_TEMPLATE.format(
            project=PROJECT,
            article=quote(article, safe=""),
            start=start.strftime("%Y%m%d"),
            end=end.strftime("%Y%m%d"),
        )
        payload = get_json(url)
        items = payload.get("items") or []
        if not items:
            continue
        timestamps = [dt.datetime.strptime(i["timestamp"], "%Y%m%d%H") for i in items]
        values = [float(i["views"]) for i in items]
        out.append(
            Series(
                series_id=f"{SOURCE_KEY}:pageviews:{article}",
                source=SOURCE_KEY,
                domain=DOMAIN,
                freq=FREQ,
                timestamps=pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)),
                values=np.array(values, dtype=float),
                metadata={"project": PROJECT, "article": article, "agent": "user"},
            )
        )
    return out
