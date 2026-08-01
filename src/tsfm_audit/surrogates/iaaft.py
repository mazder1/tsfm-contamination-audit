"""IAAFT surrogates: same spectrum, same value distribution, different instance.

The primary surrogate family. A surrogate has to preserve everything a
legitimate forecaster uses and destroy the identity a memorising model would
recall - otherwise a gap means nothing.

IAAFT (iterative amplitude-adjusted Fourier transform) preserves the power
spectrum *and* the marginal distribution, which is strictly stronger than plain
phase randomisation: the latter preserves the spectrum but leaves the values
Gaussian, so a model could distinguish real from surrogate on distribution alone
and the gap would measure nothing about memory.

What it destroys is phase structure - and with it, nonlinear dependence and
instance identity together. That conflation is why the audit needs a second
family; see :mod:`block_bootstrap`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IAAFTResult:
    surrogate: np.ndarray
    iterations: int
    converged: bool
    # Relative change in the rank ordering on the final iteration. Zero means
    # the sort order stopped moving, which is IAAFT's convergence criterion.
    final_delta: float


def iaaft(
    series: np.ndarray,
    seed: int,
    max_iter: int = 1000,
    tol: float = 0.0,
) -> IAAFTResult:
    """Generate one IAAFT surrogate of ``series``.

    The iteration alternates two projections: impose the original amplitude
    spectrum, then impose the original value distribution by rank ordering.
    Neither projection is onto a convex set, so convergence is not guaranteed in
    theory; in practice the rank ordering stops changing after tens of
    iterations, which is what ``converged`` reports.

    The distribution step runs **last**, so the returned surrogate holds the
    original values exactly - a permutation of them - and the spectrum is the
    quantity left approximate. That ordering is deliberate: an exact
    distribution is what stops a model separating real from surrogate on
    marginal statistics alone.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("IAAFT operates on a single univariate series")
    if np.isnan(values).any():
        raise ValueError("IAAFT cannot run on a series with missing values")
    n = len(values)
    if n < 8:
        raise ValueError(f"series too short for a meaningful surrogate: {n}")

    rng = np.random.default_rng(seed)

    target_amplitudes = np.abs(np.fft.rfft(values))
    sorted_values = np.sort(values)

    # Start from a random permutation: same distribution, spectrum destroyed.
    current = rng.permutation(values)
    previous_ranks = np.argsort(np.argsort(current))

    converged = False
    iterations = 0
    delta = 1.0

    for step in range(1, max_iter + 1):
        iterations = step
        # Impose the original spectrum, keeping the current phases.
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        current = np.fft.irfft(target_amplitudes * np.exp(1j * phases), n=n)

        # Impose the original distribution by rank ordering.
        ranks = np.argsort(np.argsort(current))
        current = sorted_values[ranks]

        changed = int(np.sum(ranks != previous_ranks))
        delta = changed / n
        previous_ranks = ranks
        if delta <= tol:
            converged = True
            break

    return IAAFTResult(
        surrogate=current,
        iterations=iterations,
        converged=converged,
        final_delta=float(delta),
    )


def iaaft_ensemble(
    series: np.ndarray,
    seeds: list[int],
    max_iter: int = 1000,
) -> np.ndarray:
    """Generate one surrogate per seed, shape ``(len(seeds), len(series))``.

    Surrogates are never stored - only the seeds that regenerate them - so this
    is called on demand rather than cached. See ``config.derive_seed``.
    """
    return np.stack([iaaft(series, seed=s, max_iter=max_iter).surrogate for s in seeds])
