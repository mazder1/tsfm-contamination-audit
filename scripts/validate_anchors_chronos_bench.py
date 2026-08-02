"""Option B: validate our classical models against the Chronos paper's Table 10.

Amazon published per-dataset MASE for AutoETS/AutoTheta/AutoARIMA with the full
procedure: StatsForecast, default hyperparameters, stated context caps, released
datasets, released protocol - which we already reproduce exactly (seasonal naive
to 0.024%). Matching here vindicates our implementations against an independent,
properly published reference.

Column mapping into Table 10 was verified by data anchors: the Chronos-T5(Base)
column matches our reproduced 42.68503 on covid, the SeasonalNaive column matches
our Phase 1 numbers on every row checked.

    uv run python scripts/validate_anchors_chronos_bench.py --datasets monash_m1_yearly
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
from tsfm_audit.benchmark import published  # noqa: E402

# Chronos paper, Table 10 (TMLR 10/2024), columns AutoETS/AutoTheta/AutoARIMA.
REFERENCE = {
    "monash_m1_quarterly": {"ets": 1.710, "theta": 1.683, "arima": 1.770},
    "monash_m1_yearly": {"ets": 4.110, "theta": 3.697, "arima": 3.870},
    "monash_m3_quarterly": {"ets": 1.125, "theta": 1.130, "arima": 1.419},
    "monash_m3_yearly": {"ets": 2.696, "theta": 2.613, "arima": 3.165},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["monash_m1_yearly"])
    parser.add_argument("--models", nargs="*", default=["ets", "theta", "arima"])
    args = parser.parse_args()

    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    classes = {"ets": AutoETS, "theta": AutoTheta, "arima": AutoARIMA}
    out = Path(__file__).resolve().parents[1] / "artifacts" / "anchor_validation_chronos.csv"
    rows = []
    for name in args.datasets:
        config = published.BY_NAME[name]
        windows = published.load_windows(config)
        season = get_seasonality(windows[0].freq)
        for model in args.models:
            started = time.time()
            scores = []
            for w in windows:
                past = w.past[~np.isnan(w.past)]
                if len(past) < 5:
                    continue
                fitted = classes[model](season_length=max(1, season)).fit(y=past.astype(np.float64))
                forecast = np.asarray(
                    fitted.predict(h=config.prediction_length)["mean"], dtype=float
                )
                scores.append(mase(w.target, forecast, past, season))
            ours = float(np.nanmean(scores))
            ref = REFERENCE[name][model]
            dev = 100 * (ours - ref) / ref
            rows.append(
                {
                    "dataset": name,
                    "model": model,
                    "MASE": ours,
                    "ref_MASE": ref,
                    "d_%": dev,
                    "secs": round(time.time() - started, 1),
                }
            )
            print(
                f"  {name:<24} {model:<6} ours={ours:.3f}  paper={ref:.3f}  "
                f"{dev:+.2f}%  ({rows[-1]['secs']}s)",
                flush=True,
            )
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
        subset=["dataset", "model"], keep="last"
    ).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
