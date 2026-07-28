"""One GIFT-Eval task, one model, against the published number.

The cheap integration check: prove each audited model's harness is wired up
correctly on a single task before committing to any full sweep.

Chronos goes first deliberately. Its harness is already validated against the
Chronos benchmark, so if this disagrees the fault is in the GIFT-Eval protocol
rather than the model - a distinction that vanishes if new model stacks are
introduced at the same time.

    uv run python scripts/gift_smoke.py --model chronos-base --task ett1/H/short
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
from tsfm_audit.benchmark import gift  # noqa: E402
from tsfm_audit.harness.chronos import QUANTILE_LEVELS, ChronosForecaster  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chronos-base")
    parser.add_argument("--task", default="ett1/H/short")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="gift_smoke.csv")
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if device == "cuda" else "float32"

    print(f"task {args.task}")
    windows, horizon = gift.load_task(args.task)
    season = get_seasonality(windows[0].freq)
    print(
        f"  instances={len(windows)}  horizon={horizon}  "
        f"freq={windows[0].freq}  seasonality={season}"
    )

    if args.model != "chronos-base":
        raise SystemExit(
            f"{args.model} has no harness yet - only chronos-base is wired up. "
            "Moirai needs uni2ts, Lag-Llama needs gluonts, TimesFM is a JAX/PAX "
            "checkpoint; each is a separate integration."
        )

    forecaster = ChronosForecaster(device=device, dtype=dtype, batch_size=args.batch_size)
    print(f"  model {forecaster.repo_id} @ {forecaster.revision[:8]} {device}/{dtype}\n")

    seed = config.derive_seed("gift-smoke", args.model, args.task)
    started = time.time()
    quantiles = forecaster.predict_quantiles([w.past for w in windows], horizon, seed=seed)
    median = quantiles[:, :, QUANTILE_LEVELS.index(0.5)]
    scores = [mase(w.target, median[i], w.past, season) for i, w in enumerate(windows)]
    our_mase = float(np.nanmean(scores))
    elapsed = time.time() - started

    ref = gift.published_mase(args.model, args.task)
    dev = 100 * (our_mase - ref) / ref

    print(f"  our MASE       {our_mase:.4f}")
    print(f"  published MASE {ref:.4f}")
    print(f"  deviation      {dev:+.2f}%")
    print(f"  elapsed        {elapsed:.0f}s")

    out = Path(__file__).resolve().parents[1] / "artifacts" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task": args.task,
        "model": args.model,
        "n_instances": len(windows),
        "horizon": horizon,
        "MASE": our_mase,
        "ref_MASE": ref,
        "d_MASE_%": dev,
        "device": device,
        "dtype": dtype,
        "seed": seed,
        "secs": round(elapsed, 1),
    }
    frame = pd.DataFrame([row])
    if out.exists():
        frame = pd.concat([pd.read_csv(out), frame], ignore_index=True)
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
