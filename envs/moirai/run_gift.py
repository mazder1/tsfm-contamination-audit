"""Run Moirai on one GIFT-Eval task, in Moirai's own pinned environment.

Validation target is ``moirai-1.1-R-base``, because that is the checkpoint
GIFT-Eval scored. The checkpoint under audit is ``moirai-1.0-R-base``, and it has
no published GIFT-Eval number - comparing it to the 1.1 figure would confound a
broken harness with a genuine version difference, leaving the check unable to
distinguish the two. So: validate the code path on 1.1, then point the same code
at 1.0.

Settings are GIFT-Eval's own (context 4000, patch 32, 100 samples), not this
project's audit protocol. Reproducing a published number means reproducing the
configuration it was produced under.

    uv run python run_gift.py --task ett1/H/short
    uv run python run_gift.py --task ett1/H/short --model-id Salesforce/moirai-1.0-R-base
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
from tsfm_audit.benchmark import gift  # noqa: E402

# GIFT-Eval's published Moirai configuration, from notebooks/moirai.ipynb.
GIFT_CONTEXT = 4000
GIFT_PATCH_SIZE = 32
GIFT_NUM_SAMPLES = 100
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

VALIDATION_MODEL = "Salesforce/moirai-1.1-R-base"


def build_batch(
    histories: list[np.ndarray], context_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad or left-truncate histories to a common context window.

    Recent observations are the informative ones, so a long history is cut from
    the left and a short one padded on the left, with the padding flagged rather
    than left to look like real zeros.
    """
    batch = len(histories)
    target = np.zeros((batch, context_length, 1), dtype=np.float32)
    observed = np.zeros((batch, context_length, 1), dtype=bool)
    is_pad = np.ones((batch, context_length), dtype=bool)

    for i, history in enumerate(histories):
        values = np.asarray(history, dtype=np.float32)[-context_length:]
        n = len(values)
        target[i, -n:, 0] = np.nan_to_num(values)
        observed[i, -n:, 0] = ~np.isnan(values)
        is_pad[i, -n:] = False

    return (
        torch.from_numpy(target),
        torch.from_numpy(observed),
        torch.from_numpy(is_pad),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ett1/H/short")
    parser.add_argument("--model-id", default=VALIDATION_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=GIFT_CONTEXT)
    parser.add_argument("--out", default="gift_smoke.csv")
    args = parser.parse_args()

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"task {args.task}")
    windows, horizon = gift.load_task(args.task)
    season = get_seasonality(windows[0].freq)
    print(f"  instances={len(windows)}  horizon={horizon}  seasonality={season}")
    print(f"  model {args.model_id}  device={device}")
    print(
        f"  context={args.context_length}  patch={GIFT_PATCH_SIZE}  "
        f"samples={GIFT_NUM_SAMPLES}  batch={args.batch_size}\n"
    )

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(args.model_id),
        prediction_length=horizon,
        context_length=args.context_length,
        patch_size=GIFT_PATCH_SIZE,
        num_samples=GIFT_NUM_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    model.eval()

    seed = config.derive_seed("gift-smoke", "moirai", args.task)
    torch.manual_seed(seed)

    started = time.time()
    medians: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), args.batch_size):
            chunk = windows[start : start + args.batch_size]
            target, observed, is_pad = build_batch(
                [w.past for w in chunk], args.context_length
            )
            samples = model(
                past_target=target.to(device),
                past_observed_target=observed.to(device),
                past_is_pad=is_pad.to(device),
            )
            arr = samples.float().cpu().numpy()  # (batch, sample, time, ...)
            if arr.ndim == 4:
                arr = arr[..., 0]
            medians.append(np.quantile(arr, 0.5, axis=1))
            print(
                f"    {min(start + args.batch_size, len(windows))}/{len(windows)}",
                end="\r",
                flush=True,
            )

    median = np.concatenate(medians, axis=0)
    scores = [mase(w.target, median[i], w.past, season) for i, w in enumerate(windows)]
    our_mase = float(np.nanmean(scores))
    elapsed = time.time() - started

    try:
        ref = gift.published_mase("moirai-base", args.task)
        dev = 100 * (our_mase - ref) / ref
    except Exception:  # noqa: BLE001
        ref, dev = float("nan"), float("nan")

    print(f"\n  our MASE       {our_mase:.4f}")
    print(f"  published MASE {ref:.4f}   (moirai-1.1-R-base)")
    print(f"  deviation      {dev:+.2f}%")
    print(f"  elapsed        {elapsed:.0f}s")

    out = REPO_ROOT / "artifacts" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task": args.task,
        "model": args.model_id,
        "n_instances": len(windows),
        "horizon": horizon,
        "MASE": our_mase,
        "ref_MASE": ref,
        "d_MASE_%": dev,
        "device": device,
        "dtype": "float32",
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
