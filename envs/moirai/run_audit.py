"""Moirai on both sides of the generalization probe.

Moirai is the positive control: ten benchmark datasets are PROVEN present in its
training corpus (near-duplicate search, RMS ~0). If proven contamination yields
no margin signal, the probe cannot convict anyone - that must be known first.

  --side fresh          16 fresh segments, audit protocol (ctx 512, h 24, 10 windows)
  --side contaminated   proven-in-corpus datasets under the Chronos-benchmark protocol
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402
from tsfm_audit.benchmark import published  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402

PROVEN_CONTAMINATED = [
    "monash_m1_quarterly",
    "monash_m1_yearly",
    "m4_yearly",
    "monash_m3_quarterly",
    "monash_m3_yearly",
    "monash_fred_md",
    "monash_tourism_monthly",
    "monash_tourism_yearly",
    "monash_traffic",
    "monash_australian_electricity",
]
N_WINDOWS = 10


def build_model(horizon: int, context: int, patch):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-base"),
        prediction_length=horizon,
        context_length=context,
        patch_size=patch,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    model.eval()
    return model


def forecast_batch(model, pasts, context, device):
    batch = len(pasts)
    target = np.zeros((batch, context, 1), dtype=np.float32)
    observed = np.zeros((batch, context, 1), dtype=bool)
    pad = np.ones((batch, context), dtype=bool)
    for i, p in enumerate(pasts):
        v = np.asarray(p, dtype=np.float32)[-context:]
        target[i, -len(v) :, 0] = np.nan_to_num(v)
        observed[i, -len(v) :, 0] = ~np.isnan(v)
        pad[i, -len(v) :] = False
    with torch.no_grad():
        s = model(
            past_target=torch.from_numpy(target).to(device),
            past_observed_target=torch.from_numpy(observed).to(device),
            past_is_pad=torch.from_numpy(pad).to(device),
        )
    arr = s.float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[..., 0]
    return np.quantile(arr, 0.5, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["fresh", "contaminated"], default="fresh")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = REPO_ROOT / "artifacts" / (
        "fresh_eval.csv" if args.side == "fresh" else "contaminated_eval.csv"
    )

    if args.side == "fresh":
        H, C = config.EVAL_HORIZON, config.EVAL_CONTEXT
        model = build_model(H, C, 32).to(device)
        snapshot = sorted((REPO_ROOT / "data" / "fresh").glob("fresh_*.parquet"))[-1]
        segments = [
            s
            for series in load_series(snapshot)
            for s in split_at_gaps(series, min_length=config.min_usable_segment_length())
        ]
        for seg in segments:
            values = np.asarray(seg.values, dtype=float)
            season = get_seasonality(seg.freq)
            pasts, targets = [], []
            for k in range(N_WINDOWS):
                end = len(values) - k * H
                past, tgt = values[: end - H][-C:], values[end - H : end]
                if len(past) < C or len(tgt) < H:
                    break
                pasts.append(past)
                targets.append(tgt)
            started = time.time()
            med = forecast_batch(model, pasts, C, device)
            scores = [mase(targets[i], med[i][:H], pasts[i], season) for i in range(len(targets))]
            row = {
                "model": "moirai-base",
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
            print(f"  {seg.series_id:<46} MASE={row['MASE']:.4f}", flush=True)
    else:
        for name in PROVEN_CONTAMINATED:
            cfg = published.BY_NAME[name]
            windows = published.load_windows(cfg)
            season = get_seasonality(windows[0].freq)
            C = min(512, max(len(w.past) for w in windows))
            patch = 32 if season >= 24 else 8
            model = build_model(cfg.prediction_length, C, patch).to(device)
            started = time.time()
            scores = []
            for i in range(0, len(windows), 64):
                chunk = windows[i : i + 64]
                med = forecast_batch(model, [w.past for w in chunk], C, device)
                scores += [
                    mase(w.target, med[j][: cfg.prediction_length], w.past, season)
                    for j, w in enumerate(chunk)
                ]
            row = {
                "model": "moirai-base",
                "dataset": name,
                "n_series": len(windows),
                "MASE": float(np.nanmean(scores)),
                "secs": round(time.time() - started, 1),
            }
            existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
            pd.concat([existing, pd.DataFrame([row])], ignore_index=True).drop_duplicates(
                subset=["model", "dataset"], keep="last"
            ).to_csv(out, index=False)
            print(
                f"  {name:<32} MASE={row['MASE']:.4f} ({row['secs']}s)", flush=True
            )
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
