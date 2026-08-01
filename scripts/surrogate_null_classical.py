"""Trained, specialized, non-audited models against the surrogates.

The test as actually specified: take models that are NOT zero-shot and NOT under
audit - ARIMA and ETS, each fitted to the series in front of it - run them on
real data they should be good at, then on the IAAFT and block-bootstrap versions
of the same data, and see whether performance degrades.

These models cannot have memorised anything: they have no pretraining corpus,
and they are fit fresh on every series they forecast. If they do worse on the
surrogates than on the real series, the surrogates are destroying structure
that a legitimate, specialized forecaster uses. No memory anywhere in the loop.

    uv run python scripts/surrogate_null_classical.py
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
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402
from tsfm_audit.surrogates.block_bootstrap import (  # noqa: E402
    block_bootstrap_ensemble,
    suggest_block_length,
)
from tsfm_audit.surrogates.iaaft import iaaft_ensemble  # noqa: E402

HORIZON = config.EVAL_HORIZON
CONTEXT = config.EVAL_CONTEXT


def fit_and_score(values: np.ndarray, season: int, model_name: str) -> float:
    """Fit one classical model on the history, forecast the horizon, return MASE.

    The model is trained fresh on each series it sees - which is the point.
    """
    from statsforecast.models import AutoARIMA, AutoETS

    past, target = values[:-HORIZON], values[-HORIZON:]
    season_arg = max(1, season)
    model = (
        AutoARIMA(season_length=season_arg)
        if model_name == "arima"
        else AutoETS(season_length=season_arg)
    )
    fitted = model.fit(y=past.astype(np.float64))
    forecast = fitted.predict(h=HORIZON)["mean"]
    return mase(target, np.asarray(forecast, dtype=float), past, season)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-surrogates", type=int, default=20)
    parser.add_argument("--models", nargs="*", default=["ets", "arima"])
    args = parser.parse_args()

    snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    segments = [
        seg
        for s in load_series(snapshot)
        for seg in split_at_gaps(s, min_length=config.min_usable_segment_length())
    ]
    chosen, seen = [], set()
    for seg in segments:
        if seg.domain not in seen:
            chosen.append(seg)
            seen.add(seg.domain)

    rows = []
    for seg in chosen:
        values = np.asarray(seg.values, dtype=float)[-(CONTEXT + HORIZON) :]
        season = get_seasonality(seg.freq)
        block = suggest_block_length(values, season_length=season)

        iaaft_fakes = iaaft_ensemble(
            values,
            [config.derive_seed("iaaft", seg.series_id, i) for i in range(args.n_surrogates)],
        )
        block_fakes = block_bootstrap_ensemble(
            values,
            [config.derive_seed("block", seg.series_id, i) for i in range(args.n_surrogates)],
            block_length=block,
        )

        for model_name in args.models:
            started = time.time()
            real = fit_and_score(values, season, model_name)
            iaaft_scores = [fit_and_score(s, season, model_name) for s in iaaft_fakes]
            block_scores = [fit_and_score(s, season, model_name) for s in block_fakes]
            mi = float(np.nanmedian(iaaft_scores))
            mb = float(np.nanmedian(block_scores))
            elapsed = time.time() - started

            rows.append(
                {
                    "series_id": seg.series_id,
                    "domain": seg.domain,
                    "model": model_name,
                    "real_MASE": real,
                    "iaaft_median": mi,
                    "block_median": mb,
                    "iaaft_gap_%": 100 * (mi - real) / mi,
                    "block_gap_%": 100 * (mb - real) / mb,
                    "secs": round(elapsed, 1),
                }
            )
            print(
                f"  {seg.series_id:<46} {model_name:<6} real={real:.3f}  "
                f"iaaft={mi:.3f} ({rows[-1]['iaaft_gap_%']:+.1f}%)  "
                f"block={mb:.3f} ({rows[-1]['block_gap_%']:+.1f}%)  {elapsed:.0f}s",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "surrogate_null_classical.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
