"""Shared HTTP helper: one polite, retrying session for every source."""

from __future__ import annotations

import time

import requests

# Wikimedia requires a descriptive User-Agent and will 403 generic ones.
USER_AGENT = (
    "tsfm-audit/0.1 (research; contamination audit of time-series foundation models; "
    "https://github.com/)"
)

DEFAULT_TIMEOUT = 60


def get_json(
    url: str,
    params: dict | None = None,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """GET a URL and parse JSON, retrying on transient failures."""
    return _request(url, params, retries=retries, backoff=backoff, timeout=timeout).json()


def get_text(
    url: str,
    params: dict | None = None,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """GET a URL and return the body as text."""
    return _request(url, params, retries=retries, backoff=backoff, timeout=timeout).text


def _request(
    url: str,
    params: dict | None,
    *,
    retries: int,
    backoff: float,
    timeout: int,
) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json, */*"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            status = getattr(exc.response, "status_code", None)
            # Client errors other than rate-limiting will not fix themselves.
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts") from last
