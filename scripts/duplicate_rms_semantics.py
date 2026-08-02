"""Prove what RMS=0 does and does not mean, on the real M1 data.

Two halves, as specified in review:
  1. Transform a real benchmark series in the ways RMS=0 is claimed to be blind
     to (rescale, shift). RMS must stay exactly 0.
  2. Feed it lookalikes - noisy copies, smoothed copies, off-by-one copies, and
     genuinely different series from the same dataset. RMS must NOT be 0.

If half 1 holds and half 2 holds, then RMS=0 separates 'same numbers up to
units' from 'merely similar', on this data, empirically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.probes.duplicates import mass_distance  # noqa: E402


def rms_best(query: np.ndarray, candidate: np.ndarray) -> float:
    profile = mass_distance(query, candidate)
    return float(profile.min() / np.sqrt(len(query))) if profile.size else float("nan")


def main() -> int:
    import datasets

    bench = datasets.load_dataset(
        "autogluon/chronos_datasets", "monash_m1_quarterly", split="train"
    )
    bench.set_format("numpy")
    a = np.asarray(bench[0]["target"], dtype=float)  # the series behind match #0
    b = np.asarray(bench[1]["target"], dtype=float)  # a different real series
    rng = np.random.default_rng(0)

    print("HALF 1 - transforms that must give RMS = 0 (identical up to units):")
    for label, t in [
        ("identity", a),
        ("x 1000 (unit change)", a * 1000.0),
        ("+ 500 (offset)", a + 500.0),
        ("x 0.001 - 77", a * 0.001 - 77.0),
    ]:
        print(f"  {label:<22} rms = {rms_best(a, t):.10f}")

    print("\nHALF 2 - lookalikes that must NOT give RMS = 0:")
    smoothed = np.convolve(a, np.ones(3) / 3, mode="same")
    for label, t in [
        ("+0.1% noise", a + rng.normal(0, 0.001 * a.std(), len(a))),
        ("+1% noise", a + rng.normal(0, 0.01 * a.std(), len(a))),
        ("3-point smoothing", smoothed),
        ("off by one step", np.roll(a, 1)),
        ("negated (a<0)", -a),
        ("different M1 series", b),
    ]:
        query = a[: min(len(a), len(t))]
        print(f"  {label:<22} rms = {rms_best(query, t):.10f}")

    print("\nthreshold for declaring a match: 0.05")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
