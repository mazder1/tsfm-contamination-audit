"""Near-duplicate search: find benchmark series inside a public training corpus.

The direct probe. No surrogates, no statistics about behaviour - a match is an
address: this training series, this offset, contains the benchmark series.

Criteria are pre-registered in PLAN.md Phase 6 and fixed before any corpus
content was read: z-normalized windows of W = min(len(query), 256), match at
per-point RMS <= 0.05, near-miss band to 0.25 reported but never auto-declared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MATCH_RMS = 0.05
NEAR_MISS_RMS = 0.25
MAX_WINDOW = 256


def _sliding_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of every length-``window`` subsequence, via cumulative sums."""
    cumsum = np.concatenate(([0.0], np.cumsum(values)))
    cumsq = np.concatenate(([0.0], np.cumsum(values**2)))
    n = len(values) - window + 1
    means = (cumsum[window:] - cumsum[:-window]) / window
    variances = (cumsq[window:] - cumsq[:-window]) / window - means**2
    return means[:n], np.sqrt(np.maximum(variances[:n], 1e-18))


def mass_distance(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Z-normalized Euclidean distance from ``query`` to every window of ``candidate``.

    MASS: the dot products of the z-normalized query against every candidate
    window come from one FFT convolution, so the whole profile costs O(n log n).
    """
    query = np.asarray(query, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    w, n = len(query), len(candidate)
    if w > n:
        return np.empty(0)

    q = (query - query.mean()) / (query.std() or 1.0)
    size = 1 << int(np.ceil(np.log2(n + w)))
    dots = np.fft.irfft(np.fft.rfft(candidate, size) * np.fft.rfft(q[::-1], size), size)[w - 1 : n]

    means, stds = _sliding_mean_std(candidate, w)
    # For a z-normalized query, dist^2 = 2w(1 - dot/(w*std)) after centring.
    correlation = (dots - means * q.sum()) / (w * stds)
    dist_sq = 2.0 * w * np.clip(1.0 - correlation, 0.0, 4.0)
    return np.sqrt(dist_sq)


def find_matches_with_gaps(query: np.ndarray, candidate: np.ndarray) -> list[Match]:
    """NaN-aware search: split the candidate at missing values, search each run.

    A NaN anywhere in a window poisons the FFT sums and reads as a silent
    no-match, so a series with holes was unmatchable even where 95% of it is a
    verbatim copy. Splitting into gap-free runs - the same policy the fresh
    benchmark uses - makes every clean stretch searchable, with offsets mapped
    back to positions in the original series.
    """
    candidate = np.asarray(candidate, dtype=float)
    finite = np.isfinite(candidate)
    if finite.all():
        return find_matches(query, candidate)

    out: list[Match] = []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], finite.view(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        for m in find_matches(query, candidate[start:stop]):
            out.append(Match(offset=m.offset + int(start), rms=m.rms, kind=m.kind))
    return sorted(out, key=lambda m: m.rms)


@dataclass
class Match:
    offset: int
    rms: float
    kind: str  # "match" | "near-miss"


def find_matches(query: np.ndarray, candidate: np.ndarray) -> list[Match]:
    """All pre-registered matches and near-misses of ``query`` in ``candidate``.

    The query is truncated to its first MAX_WINDOW points - identity is
    established by any sufficiently long stretch, and a fixed window keeps the
    criterion identical across queries of different lengths.
    """
    query = np.asarray(query, dtype=float)
    query = query[: min(len(query), MAX_WINDOW)]
    if len(query) < 16 or np.std(query) == 0:
        return []

    profile = mass_distance(query, candidate)
    if profile.size == 0:
        return []
    rms = profile / np.sqrt(len(query))

    out: list[Match] = []
    order = np.argsort(rms)
    taken: list[int] = []
    for index in order:
        if rms[index] > NEAR_MISS_RMS:
            break
        # Suppress overlapping hits: report the best offset per neighbourhood.
        if any(abs(index - t) < len(query) for t in taken):
            continue
        taken.append(int(index))
        out.append(
            Match(
                offset=int(index),
                rms=float(rms[index]),
                kind="match" if rms[index] <= MATCH_RMS else "near-miss",
            )
        )
    return out
