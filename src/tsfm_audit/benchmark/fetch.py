"""Fetch the fresh benchmark and write a provenance-stamped snapshot.

Usage::

    uv run python -m tsfm_audit.benchmark.fetch
    uv run python -m tsfm_audit.benchmark.fetch --sources open_meteo wikipedia

The guarantee this module enforces: every observation admitted to the fresh
benchmark is timestamped after :data:`config.FRESH_BENCHMARK_START`, which
post-dates every audited checkpoint. Anything earlier is dropped, loudly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__, config
from ..series import Series, file_sha256, save_series, series_summary, write_manifest
from .sources import FETCHERS

log = logging.getLogger("tsfm_audit.fetch")


def enforce_admissibility(series: list[Series], cutoff: dt.date) -> tuple[list[Series], int]:
    """Drop observations at or before ``cutoff``. Returns (kept, n_dropped).

    This is the load-bearing function of the whole fresh benchmark. If it is
    wrong, the benchmark is contaminated and every clean score is meaningless.
    """
    boundary = pd.Timestamp(cutoff, tz="UTC")
    kept: list[Series] = []
    dropped = 0

    for s in series:
        stamps = s.timestamps
        if stamps.tz is None:
            stamps = stamps.tz_localize("UTC")
        mask = stamps >= boundary
        dropped += int((~mask).sum())
        if not mask.any():
            log.warning("%s: no admissible observations, dropping series", s.series_id)
            continue
        kept.append(
            Series(
                series_id=s.series_id,
                source=s.source,
                domain=s.domain,
                freq=s.freq,
                timestamps=pd.DatetimeIndex(stamps[mask]),
                values=np.asarray(s.values)[mask],
                metadata=s.metadata,
            )
        )
    return kept, dropped


def fetch_all(
    start: dt.date,
    end: dt.date,
    source_keys: list[str] | None = None,
) -> tuple[list[Series], dict]:
    """Run every requested source. A source that fails is recorded, not fatal."""
    keys = source_keys or list(FETCHERS)
    collected: list[Series] = []
    status: dict[str, dict] = {}

    for key in keys:
        fetcher = FETCHERS.get(key)
        if fetcher is None:
            raise KeyError(f"unknown source {key!r}; known: {sorted(FETCHERS)}")
        log.info("fetching %s …", key)
        try:
            series = fetcher(start, end)
        except Exception as exc:  # noqa: BLE001 - one bad source must not sink the run
            log.warning("%s failed: %s", key, exc)
            status[key] = {"ok": False, "error": str(exc), "n_series": 0}
            continue
        collected.extend(series)
        status[key] = {"ok": True, "n_series": len(series)}
        log.info("%s: %d series", key, len(series))

    return collected, status


def run(
    start: dt.date | None = None,
    end: dt.date | None = None,
    source_keys: list[str] | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Fetch, validate, and write one snapshot. Returns the manifest path."""
    start = start or config.FRESH_BENCHMARK_START
    end = end or config.fresh_benchmark_end()
    out_dir = out_dir or config.FRESH_DIR

    if start < config.FRESH_BENCHMARK_START:
        raise ValueError(
            f"start {start} precedes the pre-registered cutoff "
            f"{config.FRESH_BENCHMARK_START}; refusing to fetch contaminable data"
        )

    fetched_at = dt.datetime.now(dt.UTC)
    series, status = fetch_all(start, end, source_keys)
    if not series:
        raise RuntimeError("no series fetched from any source")

    series, n_dropped = enforce_admissibility(series, config.FRESH_BENCHMARK_START)
    if n_dropped:
        log.warning("dropped %d observations predating the cutoff", n_dropped)

    stamp = fetched_at.strftime("%Y%m%d")
    data_path = out_dir / f"fresh_{stamp}.parquet"
    save_series(series, data_path)

    manifest = {
        "schema_version": 1,
        "tsfm_audit_version": __version__,
        "fetched_at_utc": fetched_at.isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "admissibility_cutoff": config.FRESH_BENCHMARK_START.isoformat(),
        "cutoff_rationale": (
            "Latest audited model released "
            f"{config.latest_model_release().isoformat()}; buffered to 2025-01-01 "
            "because release date is not data cutoff."
        ),
        "observations_dropped_before_cutoff": n_dropped,
        "sources": status,
        "n_series": len(series),
        "n_observations": int(sum(len(s) for s in series)),
        "data_file": data_path.name,
        "data_sha256": file_sha256(data_path),
        "series": series_summary(series),
    }
    manifest_path = out_dir / f"fresh_{stamp}.manifest.json"
    write_manifest(manifest_path, manifest)

    log.info(
        "wrote %d series / %d observations to %s",
        len(series),
        manifest["n_observations"],
        data_path,
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the fresh benchmark snapshot.")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=None)
    parser.add_argument("--sources", nargs="*", default=None, choices=list(FETCHERS))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    manifest_path = run(args.start, args.end, args.sources, args.out_dir)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
