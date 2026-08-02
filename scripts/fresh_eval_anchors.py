"""Generalization probe, fresh side, anchors: 4 validated classical models on the
fresh 2025-26 benchmark. 10 rolling windows per segment, context 512, horizon 24
- the audit protocol. Results feed the old-vs-fresh margin comparison.

    uv run python scripts/fresh_eval_anchors.py --models naive theta ets
    uv run python scripts/fresh_eval_anchors.py --models arima   # slow, CPU-hours
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
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402

N_WINDOWS = 10


def forecast(model, past, horizon, season):
    if model == "naive":
        return seasonal_naive_forecast(past, horizon, season)
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    cls = {"ets": AutoETS, "theta": AutoTheta, "arima": AutoARIMA}[model]
    fitted = cls(season_length=max(1, season)).fit(y=past.astype(np.float64))
    return np.asarray(fitted.predict(h=horizon)["mean"], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["naive", "theta", "ets"])
    args = parser.parse_args()

    snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    segments = [
        s
        for series in load_series(snapshot)
        for s in split_at_gaps(series, min_length=config.min_usable_segment_length())
    ]
    H, C = config.EVAL_HORIZON, config.EVAL_CONTEXT
    out = Path(__file__).resolve().parents[1] / "artifacts" / "fresh_eval.csv"
    rows = []
    for model in args.models:
        for seg in segments:
            values = np.asarray(seg.values, dtype=float)
            season = get_seasonality(seg.freq)
            scores = []
            started = time.time()
            for k in range(N_WINDOWS):
                end = len(values) - k * H
                past, target = values[: end - H][-C:], values[end - H : end]
                if len(past) < C or len(target) < H:
                    break
                scores.append(mase(target, forecast(model, past, H, season), past, season))
            rows.append(
                {
                    "model": model,
                    "series_id": seg.series_id,
                    "domain": seg.domain,
                    "n_windows": len(scores),
                    "MASE": float(np.nanmean(scores)),
                    "secs": round(time.time() - started, 1),
                }
            )
            print(
                f"  {model:<6} {seg.series_id:<46} MASE={rows[-1]['MASE']:.4f} "
                f"({rows[-1]['secs']}s)",
                flush=True,
            )
            existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
            pd.concat([existing, pd.DataFrame(rows[-1:])], ignore_index=True).drop_duplicates(
                subset=["model", "series_id"], keep="last"
            ).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
