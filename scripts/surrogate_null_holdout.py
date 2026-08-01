"""Out-of-sample test of the two-group hypothesis, predictions registered first.

The linear-vs-shape split was discovered post hoc: the same runs that suggested
it were used to confirm it, and one of the six forecasters involved (Chronos) is
a model under audit. Review refused that, correctly. So: two NEW forecasters
from mechanism families none of the six belong to, with predictions committed
before the run.

  harmonic  - OLS on sines/cosines at seasonal harmonics plus trend. Pure
              frequency-domain statistics. PREDICTION: fakes easier or neutral,
              gap <= 0.
  swa       - seasonal window average: the forecast is the mean of the last few
              seasonal cycles' shapes. Shape-based via averaging. PREDICTION:
              fakes harder, gap > +10%.

Both are fit per series. Nothing audited anywhere in the loop.

    uv run python scripts/surrogate_null_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402
from tsfm_audit.surrogates.block_bootstrap import (  # noqa: E402
    block_bootstrap_ensemble,
    suggest_block_length,
)
from tsfm_audit.surrogates.iaaft import iaaft_ensemble  # noqa: E402

HORIZON = config.EVAL_HORIZON
CONTEXT = config.EVAL_CONTEXT
N_SURR = 20
N_HARMONICS = 8
SWA_WINDOW = 4  # average the last 4 cycles


def harmonic_forecast(past: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """OLS on [trend, sin, cos] at the seasonal harmonics. Pure spectral fit."""
    n = len(past)
    t = np.arange(n + horizon, dtype=float)
    columns = [np.ones_like(t), t]
    for k in range(1, N_HARMONICS + 1):
        columns.append(np.sin(2 * np.pi * k * t / season))
        columns.append(np.cos(2 * np.pi * k * t / season))
    design = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(design[:n], past, rcond=None)
    return design[n:] @ coef


def swa_forecast(past: np.ndarray, horizon: int, season: int) -> np.ndarray:
    """Mean shape of the last SWA_WINDOW full cycles, repeated forward."""
    cycles = past[-season * SWA_WINDOW :].reshape(SWA_WINDOW, season)
    mean_cycle = cycles.mean(axis=0)
    reps = int(np.ceil(horizon / season))
    return np.tile(mean_cycle, reps)[:horizon]


def score(values: np.ndarray, season: int, forecaster) -> float:
    past, target = values[:-HORIZON], values[-HORIZON:]
    return mase(target, forecaster(past, HORIZON, season), past, season)


def main() -> int:
    snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    segments = [
        seg
        for s in load_series(snapshot)
        for seg in split_at_gaps(s, min_length=config.min_usable_segment_length())
    ]
    chosen, seen = [], set()
    for seg in segments:
        if seg.domain not in seen and seg.domain != "web_traffic":
            # web traffic already shown spike-dominated and uninformative here
            chosen.append(seg)
            seen.add(seg.domain)

    forecasters = {"harmonic": harmonic_forecast, "swa": swa_forecast}
    rows = []
    for seg in chosen:
        values = np.asarray(seg.values, dtype=float)[-(CONTEXT + HORIZON) :]
        season = get_seasonality(seg.freq)
        block = suggest_block_length(values, season_length=season)
        iaaft_fakes = iaaft_ensemble(
            values, [config.derive_seed("iaaft", seg.series_id, i) for i in range(N_SURR)]
        )
        block_fakes = block_bootstrap_ensemble(
            values,
            [config.derive_seed("block", seg.series_id, i) for i in range(N_SURR)],
            block_length=block,
        )
        for name, forecaster in forecasters.items():
            real = score(values, season, forecaster)
            mi = float(np.nanmedian([score(s, season, forecaster) for s in iaaft_fakes]))
            mb = float(np.nanmedian([score(s, season, forecaster) for s in block_fakes]))
            rows.append(
                {
                    "series_id": seg.series_id,
                    "model": name,
                    "real_MASE": real,
                    "iaaft_median": mi,
                    "block_median": mb,
                    "iaaft_gap_%": 100 * (mi - real) / mi,
                    "block_gap_%": 100 * (mb - real) / mb,
                }
            )
            print(
                f"  {seg.series_id:<46} {name:<9} real={real:.3f}  "
                f"iaaft={mi:.3f} ({rows[-1]['iaaft_gap_%']:+.1f}%)  "
                f"block={mb:.3f} ({rows[-1]['block_gap_%']:+.1f}%)",
                flush=True,
            )

    out = Path(__file__).resolve().parents[1] / "artifacts" / "surrogate_null_holdout.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
