"""Block bootstrap surrogates: keep local nonlinear dynamics, destroy identity.

The second surrogate family, and the reason the audit can tell memorisation
apart from skill.

IAAFT destroys nonlinear structure and instance identity at the same time, so a
gap under IAAFT alone is ambiguous - the model might be exploiting nonlinearity
rather than remembering. Resampling contiguous blocks preserves whatever
happens *inside* a block, nonlinear dependence included, while destroying the
long-range arrangement that makes a series that particular series.

    Gap under both families  -> memorisation
    Gap under IAAFT only     -> nonlinear skill, not a finding
    No gap                   -> clean
"""

from __future__ import annotations

import numpy as np


def moving_block_bootstrap(
    series: np.ndarray,
    seed: int,
    block_length: int,
) -> np.ndarray:
    """Resample overlapping blocks with replacement to the original length.

    Overlapping (moving) blocks rather than disjoint ones: with disjoint blocks
    every surrogate is built from the same few segments, which both reduces
    variety across an ensemble and makes the surrogate easier to recognise as a
    rearrangement of the original.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("block bootstrap operates on a single univariate series")
    if np.isnan(values).any():
        raise ValueError("block bootstrap cannot run on a series with missing values")
    n = len(values)
    if not 1 <= block_length <= n:
        raise ValueError(f"block_length must be in [1, {n}], got {block_length}")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    out = np.concatenate([values[s : s + block_length] for s in starts])
    return out[:n]


def block_bootstrap_ensemble(
    series: np.ndarray,
    seeds: list[int],
    block_length: int,
) -> np.ndarray:
    """One surrogate per seed, shape ``(len(seeds), len(series))``."""
    return np.stack(
        [moving_block_bootstrap(series, seed=s, block_length=block_length) for s in seeds]
    )


def suggest_block_length(series: np.ndarray, season_length: int | None = None) -> int:
    """Block length from a stated criterion rather than a tuned one.

    Two competing requirements: long enough to contain the dependence we mean to
    preserve, short enough that the arrangement of blocks is genuinely destroyed.

    The rule is the larger of one seasonal period - so a block holds a whole
    cycle rather than a fragment of one - and ``n**(1/3)``, the standard
    asymptotic rate for block bootstrap under general weak dependence. Neither
    term is chosen by looking at results, which is the point: PLAN.md commits to
    deciding this by a stated criterion rather than by which value gives the
    nicer answer.
    """
    n = len(series)
    asymptotic = max(1, int(round(n ** (1 / 3))))
    if season_length and season_length > 1:
        return int(min(max(asymptotic, season_length), n))
    return int(min(asymptotic, n))
