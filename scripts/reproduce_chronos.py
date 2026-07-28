"""Phase 1: reproduce the published Chronos zero-shot results.

The soft gate. Chronos samples and the reference evaluation sets no seed, so
exact agreement is unavailable; the tolerance is defined in PLAN.md and was
fixed before any full-benchmark number was produced.

    uv run python scripts/reproduce_chronos.py --datasets ETTh
    uv run python scripts/reproduce_chronos.py                 # 23 smaller sets
    uv run python scripts/reproduce_chronos.py --only-large    # dominick, m5, m4_*
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality, mase, weighted_quantile_loss  # noqa: E402
from tsfm_audit.benchmark import published  # noqa: E402
from tsfm_audit.harness.chronos import QUANTILE_LEVELS, ChronosForecaster  # noqa: E402

# Pre-registered: seeds are spent where noise is large, i.e. where series are few.
SEEDS_WHEN_SMALL = 5
SMALL_DATASET_SERIES = 500
NOISE_CONSTANT = 18.5  # single-series run-to-run wobble, in percent
CENTRE_TOLERANCE = 2.5  # absorbs the unknown-sign checkpoint offset


def band(n_series: int) -> float:
    """Half-width of the pass band, in percent, for a dataset of this size."""
    return CENTRE_TOLERANCE + 3.0 * NOISE_CONSTANT / np.sqrt(max(n_series, 1))


def score_once(forecaster, config_, windows, season, seed) -> dict:
    quantiles = forecaster.predict_quantiles(
        [w.past for w in windows], config_.prediction_length, seed=seed
    )
    median = quantiles[:, :, QUANTILE_LEVELS.index(0.5)]
    scores = [mase(w.target, median[i], w.past, season) for i, w in enumerate(windows)]
    return {
        "MASE": float(np.nanmean(scores)),
        "WQL": weighted_quantile_loss(
            [w.target for w in windows], list(quantiles), QUANTILE_LEVELS
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--only-large", action="store_true")
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--out", default="chronos_reproduction.csv")
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto")
    parser.add_argument("--dtype", default=None, help="default bfloat16 on GPU, float32 on CPU")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # bfloat16 is the fast path on Ampere and costs at most 0.15% on MASE
    # (measured); on CPU it is 8x slower than float32, so the default flips.
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")

    if args.datasets:
        configs = [published.BY_NAME[n] for n in args.datasets]
    elif args.only_large:
        configs = [c for c in published.ZERO_SHOT if c.name in published.LARGE]
    else:
        configs = [
            c for c in published.ZERO_SHOT if args.include_large or c.name not in published.LARGE
        ]

    out = Path(__file__).resolve().parents[1] / "artifacts" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    reference = published.load_published_results("chronos-t5-base")
    forecaster = ChronosForecaster(device=device, dtype=dtype, batch_size=args.batch_size)
    print(
        f"model {forecaster.repo_id} @ {forecaster.revision[:8]}  "
        f"device={device}  dtype={dtype}  batch={args.batch_size}\n"
    )

    rows = []
    for config_ in configs:
        started = time.time()
        windows = published.load_windows(config_)
        season = get_seasonality(windows[0].freq)
        n = len(windows)
        seeds = [
            config.derive_seed("chronos-phase1", config_.name, i)
            for i in range(SEEDS_WHEN_SMALL if n <= SMALL_DATASET_SERIES else 1)
        ]

        runs = [score_once(forecaster, config_, windows, season, s) for s in seeds]
        our_mase = float(np.mean([r["MASE"] for r in runs]))
        our_wql = float(np.mean([r["WQL"] for r in runs]))

        ref = reference.loc[config_.name]
        dev = 100 * (our_mase - float(ref["MASE"])) / float(ref["MASE"])
        allowed = band(n)
        elapsed = time.time() - started

        rows.append(
            {
                "dataset": config_.name,
                "n_series": n,
                "horizon": config_.prediction_length,
                "n_seeds": len(seeds),
                "MASE": our_mase,
                "ref_MASE": float(ref["MASE"]),
                "d_MASE_%": dev,
                "band_%": allowed,
                "pass": abs(dev) <= allowed,
                "WQL": our_wql,
                "ref_WQL": float(ref["WQL"]),
                "d_WQL_%": 100 * (our_wql - float(ref["WQL"])) / float(ref["WQL"]),
                "device": device,
                "dtype": dtype,
                "secs": round(elapsed, 1),
                "secs_per_series": round(elapsed / max(n, 1) / len(seeds), 4),
            }
        )
        verdict = "pass" if rows[-1]["pass"] else "FAIL"
        print(
            f"  {config_.name:<32} n={n:<6} h={config_.prediction_length:<3} "
            f"MASE={our_mase:.4f} (ref {float(ref['MASE']):.4f}, {dev:+.2f}%) "
            f"band +/-{allowed:.2f}%  {verdict}  {elapsed:.0f}s",
            flush=True,
        )
        # Written after every dataset, not at the end: a long unattended run
        # that dies at dataset 20 should still leave 19 results behind.
        pd.DataFrame(rows).to_csv(out, index=False)

    frame = pd.DataFrame(rows)

    print(f"\nwrote {out}")
    print(f"median signed deviation: {frame['d_MASE_%'].median():+.3f}%  (allowed +/-2.5%)")
    print(f"datasets passing band:   {int(frame['pass'].sum())}/{len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
