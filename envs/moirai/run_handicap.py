"""Handicap curve: Moirai on clean fresh electricity at deliberately short contexts.

Measures how much of the poor short-series contaminated ratios is explained by
short history alone. Fresh data cannot be memorised, so this curve is pure
handicap; if the contaminated short-series ratios sit ON it, handicap explains
them fully and no memorisation signal exists anywhere. Sitting clearly BELOW the
curve (better than clean-data handicap predicts) would be the first real
memorisation residue.

Patch size 8 below context 64, else 32 - noted as a protocol seam.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_audit import build_model, forecast_batch  # noqa: E402

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402

CONTEXTS = [16, 32, 64, 128, 512]
N_WINDOWS = 10
H = 24


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snapshot = sorted((REPO_ROOT / "data" / "fresh").glob("fresh_*.parquet"))[-1]
    segments = [
        s
        for series in load_series(snapshot)
        for s in split_at_gaps(series, min_length=config.min_usable_segment_length())
        if s.domain == "electricity"
    ]
    out = REPO_ROOT / "artifacts" / "moirai_handicap_curve.csv"
    rows = []
    for context in CONTEXTS:
        patch = 8 if context < 64 else 32
        model = build_model(H, context, patch).to(device)
        for seg in segments:
            values = np.asarray(seg.values, dtype=float)
            season = get_seasonality(seg.freq)
            pasts, targets = [], []
            for k in range(N_WINDOWS):
                end = len(values) - k * H
                past, tgt = values[: end - H][-context:], values[end - H : end]
                if len(past) < context or len(tgt) < H:
                    break
                pasts.append(past)
                targets.append(tgt)
            started = time.time()
            med = forecast_batch(model, pasts, context, device)
            # MASE denominator from the SAME short past each side sees, so the
            # metric is internally consistent at every context length.
            scores = [
                mase(targets[i], med[i][:H], pasts[i], season) for i in range(len(targets))
            ]
            rows.append(
                {
                    "context": context,
                    "series_id": seg.series_id,
                    "MASE": float(np.nanmean(scores)),
                    "secs": round(time.time() - started, 1),
                }
            )
            print(
                f"  ctx={context:<4} {seg.series_id:<24} MASE={rows[-1]['MASE']:.4f}",
                flush=True,
            )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
