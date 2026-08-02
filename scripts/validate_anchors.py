"""Validate the classical anchor models against GIFT-Eval's published results.

The anchors judge the foundation models on fresh data, so they get the same
treatment the foundation models got: our implementation, their published number,
same task ett1/H/short, same protocol. An unvalidated anchor is a bent ruler.

    uv run python scripts/validate_anchors.py --models naive ets theta
    uv run python scripts/validate_anchors.py --models arima   # hours on CPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.baselines.seasonal_naive import seasonal_naive_forecast  # noqa: E402
from tsfm_audit.benchmark import gift  # noqa: E402

# Published result directories in SalesforceAIResearch/gift-eval.
PUBLISHED_DIR = {
    "naive": "seasonal_naive",
    "ets": "auto_ets",
    "theta": "auto_theta",
    "arima": "auto_arima",
}
TASK = "ett1/H/short"


def forecast(model: str, past: np.ndarray, horizon: int, season: int) -> np.ndarray:
    if model == "naive":
        return seasonal_naive_forecast(past, horizon, season)
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    cls = {"ets": AutoETS, "theta": AutoTheta, "arima": AutoARIMA}[model]
    fitted = cls(season_length=season).fit(y=past.astype(np.float64))
    return np.asarray(fitted.predict(h=horizon)["mean"], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["naive", "ets", "theta"])
    args = parser.parse_args()

    windows, horizon = gift.load_task(TASK)
    season = get_seasonality(windows[0].freq)
    print(f"{TASK}: {len(windows)} windows, horizon {horizon}", flush=True)

    out = Path(__file__).resolve().parents[1] / "artifacts" / "anchor_validation.csv"
    for model in args.models:
        started = time.time()
        scores = []
        for w in windows:
            past = w.past[-2048:]  # bound ARIMA cost; plenty for classical fits
            scores.append(mase(w.target, forecast(model, past, horizon, season), w.past, season))
        ours = float(np.nanmean(scores))
        frame = pd.read_csv(gift.RESULTS_URL.format(model=PUBLISHED_DIR[model])).set_index(
            "dataset"
        )
        ref = float(frame.loc[TASK, "eval_metrics/MASE[0.5]"])
        dev = 100 * (ours - ref) / ref
        row = {
            "model": model,
            "task": TASK,
            "MASE": ours,
            "ref_MASE": ref,
            "d_%": dev,
            "secs": round(time.time() - started, 1),
        }
        existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).drop_duplicates(
            subset=["model", "task"], keep="last"
        ).to_csv(out, index=False)
        print(
            f"  {model:<7} ours={ours:.4f}  published={ref:.4f}  {dev:+.2f}%  ({row['secs']}s)",
            flush=True,
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
