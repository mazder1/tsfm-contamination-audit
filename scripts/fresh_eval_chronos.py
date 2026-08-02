"""Generalization probe, fresh side: Chronos-base on the fresh benchmark.
Same protocol as the anchors: 10 rolling windows, context 512, horizon 24.
Appends to artifacts/fresh_eval.csv as model chronos-base.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.harness.chronos import QUANTILE_LEVELS, ChronosForecaster  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402

N_WINDOWS = 10


def main() -> int:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forecaster = ChronosForecaster(
        device=device, dtype="bfloat16" if device == "cuda" else "float32", batch_size=8
    )
    snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    segments = [
        s
        for series in load_series(snapshot)
        for s in split_at_gaps(series, min_length=config.min_usable_segment_length())
    ]
    H, C = config.EVAL_HORIZON, config.EVAL_CONTEXT
    out = Path(__file__).resolve().parents[1] / "artifacts" / "fresh_eval.csv"
    for seg in segments:
        values = np.asarray(seg.values, dtype=float)
        season = get_seasonality(seg.freq)
        pasts, targets = [], []
        for k in range(N_WINDOWS):
            end = len(values) - k * H
            past, target = values[: end - H][-C:], values[end - H : end]
            if len(past) < C or len(target) < H:
                break
            pasts.append(past)
            targets.append(target)
        started = time.time()
        seed = config.derive_seed("fresh-eval", "chronos-base", seg.series_id)
        q = forecaster.predict_quantiles(pasts, H, seed=seed)
        med = q[:, :, QUANTILE_LEVELS.index(0.5)]
        scores = [mase(targets[i], med[i], pasts[i], season) for i in range(len(targets))]
        row = {
            "model": "chronos-base",
            "series_id": seg.series_id,
            "domain": seg.domain,
            "n_windows": len(scores),
            "MASE": float(np.nanmean(scores)),
            "secs": round(time.time() - started, 1),
        }
        existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).drop_duplicates(
            subset=["model", "series_id"], keep="last"
        ).to_csv(out, index=False)
        print(f"  {seg.series_id:<46} MASE={row['MASE']:.4f} ({row['secs']}s)", flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
