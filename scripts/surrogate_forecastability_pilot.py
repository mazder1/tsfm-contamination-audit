"""Are the surrogates forecastable at all? The check that precedes every other.

Phase 3.5's null control asks whether the probe stays *silent* where contamination
is impossible. This asks something more basic and more dangerous if unanswered:
whether a model can forecast a surrogate at a comparable level to the real series
in the first place.

The two are not the same test. If IAAFT produced series with the right histogram
and the right spectrum but no forecastable structure, every model would score
catastrophically on them, the gap would be enormous, the null control would fire,
and we would know only that *something* was wrong. Looking at the raw scores says
what.

This runs on the fresh benchmark, which postdates every audited checkpoint, so
any gap here is a property of the surrogates rather than of memorisation.

No p-values, no FDR, no decision rule. Just the numbers, so they can be looked at
before anything is built on top of them.

    uv run python scripts/surrogate_forecastability_pilot.py
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
from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.baselines.seasonal_naive import seasonal_naive_forecast  # noqa: E402
from tsfm_audit.harness.chronos import QUANTILE_LEVELS, ChronosForecaster  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402
from tsfm_audit.surrogates.block_bootstrap import (  # noqa: E402
    block_bootstrap_ensemble,
    suggest_block_length,
)
from tsfm_audit.surrogates.iaaft import iaaft_ensemble  # noqa: E402

HORIZON = config.EVAL_HORIZON
CONTEXT = config.EVAL_CONTEXT


def score_series(forecaster, values, season, seed):
    """MASE of one series, forecasting the final HORIZON points from CONTEXT before."""
    past, target = values[:-HORIZON][-CONTEXT:], values[-HORIZON:]
    quantiles = forecaster.predict_quantiles([past], HORIZON, seed=seed)
    median = quantiles[0, :, QUANTILE_LEVELS.index(0.5)]
    return mase(target, median, past, season)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-surrogates", type=int, default=20)
    parser.add_argument("--max-series", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free = torch.cuda.mem_get_info()[0] // (1024 * 1024)
        print(f"gpu free at start: {free} MiB")

    snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    segments = [
        segment
        for series in load_series(snapshot)
        for segment in split_at_gaps(series, min_length=config.min_usable_segment_length())
    ]
    # One per domain, so a surrogate failure specific to one kind of data shows up.
    chosen, seen = [], set()
    for segment in segments:
        if segment.domain not in seen:
            chosen.append(segment)
            seen.add(segment.domain)
        if len(chosen) >= args.max_series:
            break

    forecaster = ChronosForecaster(
        device=device,
        dtype="bfloat16" if device == "cuda" else "float32",
        batch_size=args.batch_size,
    )
    print(f"model {forecaster.repo_id} @ {forecaster.revision[:8]}  {device}\n")

    rows = []
    for segment in chosen:
        values = np.asarray(segment.values, dtype=float)[-(CONTEXT + HORIZON) :]
        season = get_seasonality(segment.freq)
        block = suggest_block_length(values, season_length=season)
        seed = config.derive_seed("forecastability", segment.series_id)
        started = time.time()

        real = score_series(forecaster, values, season, seed)

        # Seasonal naive on the same series, as a scale reference: MASE near 1
        # means "about as good as the trivial baseline".
        past, target = values[:-HORIZON], values[-HORIZON:]
        naive = mase(target, seasonal_naive_forecast(past, HORIZON, season), past, season)

        family_scores = {}
        for family, ensemble in (
            (
                "iaaft",
                iaaft_ensemble(
                    values,
                    [
                        config.derive_seed("iaaft", segment.series_id, i)
                        for i in range(args.n_surrogates)
                    ],
                ),
            ),
            (
                "block_bootstrap",
                block_bootstrap_ensemble(
                    values,
                    [
                        config.derive_seed("block", segment.series_id, i)
                        for i in range(args.n_surrogates)
                    ],
                    block_length=block,
                ),
            ),
        ):
            scores = [score_series(forecaster, s, season, seed) for s in ensemble]
            family_scores[family] = np.array(scores, dtype=float)

        elapsed = time.time() - started
        row = {
            "series_id": segment.series_id,
            "domain": segment.domain,
            "season": season,
            "block_length": block,
            "real_MASE": real,
            "seasonal_naive_MASE": naive,
            "secs": round(elapsed, 1),
        }
        for family, scores in family_scores.items():
            row[f"{family}_median"] = float(np.nanmedian(scores))
            row[f"{family}_min"] = float(np.nanmin(scores))
            row[f"{family}_max"] = float(np.nanmax(scores))
            # Relative gap, the quantity the real probe will test. Positive means
            # the model did better on the real series than on its surrogates.
            row[f"{family}_gap_%"] = 100 * (np.nanmedian(scores) - real) / np.nanmedian(scores)
        rows.append(row)

        print(f"  {segment.series_id}  ({segment.domain}, {elapsed:.0f}s)")
        print(f"    real                {real:.4f}")
        print(f"    seasonal naive      {naive:.4f}")
        for family, scores in family_scores.items():
            print(
                f"    {family:<18} median {np.nanmedian(scores):.4f}  "
                f"range {np.nanmin(scores):.4f}-{np.nanmax(scores):.4f}  "
                f"gap {row[f'{family}_gap_%']:+.1f}%"
            )
        print(flush=True)

    frame = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "surrogate_forecastability.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
