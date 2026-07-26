"""The one series container every part of the pipeline passes around."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Series:
    """A single univariate time series with provenance attached.

    Provenance is not decoration: the whole project turns on being able to say
    exactly where a series came from and when it was published.
    """

    series_id: str
    source: str
    domain: str
    freq: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if len(self.timestamps) != len(self.values):
            raise ValueError(
                f"{self.series_id}: {len(self.timestamps)} timestamps vs {len(self.values)} values"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def content_hash(self) -> str:
        """Stable hash of the values — the cache key for forecasts."""
        return hashlib.sha256(self.values.tobytes()).hexdigest()[:16]

    @property
    def n_missing(self) -> int:
        return int(np.isnan(self.values).sum())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "series_id": self.series_id,
                "source": self.source,
                "domain": self.domain,
                "freq": self.freq,
                "timestamp": self.timestamps,
                "value": self.values,
            }
        )


def save_series(series: list[Series], path: Path) -> Path:
    """Write a set of series to one parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([s.to_frame() for s in series], ignore_index=True)
    frame.to_parquet(path, index=False)
    return path


def load_series(path: Path) -> list[Series]:
    """Read series back from a parquet file written by :func:`save_series`."""
    frame = pd.read_parquet(path)
    out: list[Series] = []
    for series_id, group in frame.groupby("series_id", sort=True):
        group = group.sort_values("timestamp")
        out.append(
            Series(
                series_id=str(series_id),
                source=str(group["source"].iloc[0]),
                domain=str(group["domain"].iloc[0]),
                freq=str(group["freq"].iloc[0]),
                timestamps=pd.DatetimeIndex(group["timestamp"]),
                values=group["value"].to_numpy(dtype=float),
            )
        )
    return out


def write_manifest(path: Path, payload: dict) -> Path:
    """Write a provenance manifest beside a data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series_summary(series: list[Series]) -> list[dict]:
    """Compact per-series description for manifests."""
    return [
        {
            "series_id": s.series_id,
            "source": s.source,
            "domain": s.domain,
            "freq": s.freq,
            "n_obs": len(s),
            "n_missing": s.n_missing,
            "start": str(s.timestamps[0]) if len(s) else None,
            "end": str(s.timestamps[-1]) if len(s) else None,
            "content_hash": s.content_hash,
            **({"metadata": s.metadata} if s.metadata else {}),
        }
        for s in series
    ]
