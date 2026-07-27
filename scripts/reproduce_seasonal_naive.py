"""Phase 1, hard gate: reproduce the published seasonal-naive zero-shot results.

No model, no GPU - pure arithmetic over the benchmark datasets. If our numbers
disagree with the reference here, our scoring is wrong and nothing downstream is
worth running.

    uv run python scripts/reproduce_seasonal_naive.py --datasets monash_cif_2016 ercot
    uv run python scripts/reproduce_seasonal_naive.py            # all but the large four
    uv run python scripts/reproduce_seasonal_naive.py --include-large
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.analysis.metrics import (  # noqa: E402
    get_seasonality,
    mase,
    weighted_quantile_loss,
)
from tsfm_audit.baselines.seasonal_naive import seasonal_naive_quantiles  # noqa: E402
from tsfm_audit.benchmark import published  # noqa: E402

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def score_dataset(config: published.BacktestConfig) -> dict:
    windows = published.load_windows(config)
    if not windows:
        return {"dataset": config.name, "error": "no windows"}

    freq = windows[0].freq
    season = get_seasonality(freq)

    per_series_mase: list[float] = []
    targets: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []

    for window in windows:
        quantiles = seasonal_naive_quantiles(
            window.past, config.prediction_length, season, QUANTILES
        )
        median = quantiles[:, QUANTILES.index(0.5)]
        per_series_mase.append(mase(window.target, median, window.past, season))
        targets.append(window.target)
        forecasts.append(quantiles)

    return {
        "dataset": config.name,
        "freq": freq,
        "season": season,
        "n_series": len(windows),
        "MASE": float(np.nanmean(per_series_mase)),
        "WQL": weighted_quantile_loss(targets, forecasts, QUANTILES),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--include-large", action="store_true")
    args = parser.parse_args()

    if args.datasets:
        configs = [published.BY_NAME[n] for n in args.datasets]
    else:
        configs = [
            c for c in published.ZERO_SHOT if args.include_large or c.name not in published.LARGE
        ]

    reference = published.load_published_results("seasonal-naive")

    rows = []
    for config in configs:
        started = time.time()
        try:
            result = score_dataset(config)
        except Exception as exc:  # noqa: BLE001 - one bad dataset must not sink the run
            print(f"  FAILED  {config.name}: {type(exc).__name__}: {exc}")
            rows.append({"dataset": config.name, "error": str(exc)[:120]})
            continue

        if config.name in reference.index:
            ref = reference.loc[config.name]
            result["ref_MASE"] = float(ref["MASE"])
            result["ref_WQL"] = float(ref["WQL"])
            result["d_MASE_%"] = 100 * (result["MASE"] - result["ref_MASE"]) / result["ref_MASE"]
            result["d_WQL_%"] = 100 * (result["WQL"] - result["ref_WQL"]) / result["ref_WQL"]
        result["secs"] = round(time.time() - started, 1)
        rows.append(result)
        print(
            f"  {config.name:<32} freq={result['freq']:<5} m={result['season']:<4} "
            f"n={result['n_series']:<7} MASE={result['MASE']:.4f} "
            f"(ref {result.get('ref_MASE', float('nan')):.4f}, "
            f"{result.get('d_MASE_%', float('nan')):+.2f}%)"
        )

    frame = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "seasonal_naive_reproduction.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")

    if "d_MASE_%" in frame:
        worst = frame["d_MASE_%"].abs().max()
        print(f"worst absolute MASE deviation: {worst:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
