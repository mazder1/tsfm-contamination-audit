"""Measure how much of a Chronos-vs-published gap is noise rather than error.

Run before setting the Phase 1 reproduction tolerance, so the tolerance is
derived from measured variance instead of invented. Two questions:

  1. how far does per-dataset MASE move across sampling seeds?
  2. does dtype (float32 on CPU vs the reference's bfloat16) shift it?

    uv run python scripts/measure_chronos_variance.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.analysis.metrics import get_seasonality, mase, weighted_quantile_loss  # noqa: E402
from tsfm_audit.benchmark import published  # noqa: E402
from tsfm_audit.harness.chronos import QUANTILE_LEVELS, ChronosForecaster  # noqa: E402

# Small enough to run many times on CPU, and spread across frequencies.
PROBE_DATASETS = ("monash_cif_2016", "ercot", "monash_m1_quarterly")
SEEDS = (0, 1, 2, 3, 4)


def score(forecaster: ChronosForecaster, config, windows, seed: int) -> dict:
    season = get_seasonality(windows[0].freq)
    quantiles = forecaster.predict_quantiles(
        [w.past for w in windows], config.prediction_length, seed=seed
    )
    median_index = QUANTILE_LEVELS.index(0.5)
    scores = [
        mase(w.target, quantiles[i, :, median_index], w.past, season) for i, w in enumerate(windows)
    ]
    return {
        "MASE": float(np.nanmean(scores)),
        "WQL": weighted_quantile_loss(
            [w.target for w in windows], list(quantiles), QUANTILE_LEVELS
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(PROBE_DATASETS))
    args = parser.parse_args()

    reference = published.load_published_results("chronos-t5-base")
    rows = []

    for dtype in ("float32", "bfloat16"):
        print(f"\n=== dtype {dtype} ===")
        forecaster = ChronosForecaster(dtype=dtype)
        for name in args.datasets:
            config = published.BY_NAME[name]
            windows = published.load_windows(config)
            for seed in SEEDS:
                started = time.time()
                result = score(forecaster, config, windows, seed)
                ref = float(reference.loc[name, "MASE"])
                rows.append(
                    {
                        "dataset": name,
                        "dtype": dtype,
                        "seed": seed,
                        "n_series": len(windows),
                        **result,
                        "ref_MASE": ref,
                        "d_MASE_%": 100 * (result["MASE"] - ref) / ref,
                        "secs": round(time.time() - started, 1),
                    }
                )
                print(
                    f"  {name:<24} seed={seed}  MASE={result['MASE']:.4f}  "
                    f"(ref {ref:.4f}, {rows[-1]['d_MASE_%']:+.2f}%)  "
                    f"{rows[-1]['secs']}s"
                )

    frame = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "chronos_variance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print("\n=== seed-to-seed spread, per dataset and dtype ===")
    summary = frame.groupby(["dataset", "dtype"])["MASE"].agg(["mean", "std", "min", "max"])
    summary["rel_sd_%"] = 100 * summary["std"] / summary["mean"]
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))

    print("\n=== deviation from published, per dataset and dtype ===")
    dev = frame.groupby(["dataset", "dtype"])["d_MASE_%"].agg(["mean", "std"])
    print(dev.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
