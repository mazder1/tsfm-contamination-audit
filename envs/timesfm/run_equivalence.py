"""TimesFM PyTorch side of the JAX-vs-PyTorch equivalence test.

Reads the frozen contexts, forecasts them, writes the point forecasts. Nothing
else - no benchmark loading, no metrics - so the two sides differ only in which
runtime produced the numbers.

    uv run python run_equivalence.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"

PYTORCH_REPO = "google/timesfm-1.0-200m-pytorch"

# TimesFM 1.0-200m architecture. Fixed by the checkpoint, not chosen by us.
NUM_LAYERS = 20
MODEL_DIMS = 1280
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
CONTEXT_LEN = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="timesfm_equivalence_inputs.npz")
    parser.add_argument("--out", default="timesfm_forecasts_pytorch.npz")
    parser.add_argument("--repo", default=PYTORCH_REPO)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import timesfm

    frozen = np.load(ARTIFACTS / args.inputs)
    contexts = frozen["contexts"]
    horizon = int(frozen["horizon"])
    print(f"loaded {contexts.shape[0]} contexts of {contexts.shape[1]}, horizon {horizon}")

    hparams = timesfm.TimesFmHparams(
        backend="gpu",
        per_core_batch_size=args.batch_size,
        horizon_len=horizon,
        context_len=CONTEXT_LEN,
        num_layers=NUM_LAYERS,
        model_dims=MODEL_DIMS,
        input_patch_len=INPUT_PATCH_LEN,
        output_patch_len=OUTPUT_PATCH_LEN,
    )
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=args.repo)

    # timesfm 1.2 exposes the PyTorch backend as a separate class; older layouts
    # select it through hparams. Try the explicit class first and say plainly
    # which path was taken, since the whole point here is knowing what ran.
    if hasattr(timesfm, "TimesFmTorch"):
        model = timesfm.TimesFmTorch(hparams=hparams, checkpoint=checkpoint)
        backend_used = "TimesFmTorch"
    else:
        model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        backend_used = "TimesFm(backend=gpu)"
    print(f"backend: {backend_used}  repo: {args.repo}")

    point_forecast, _ = model.forecast(
        [contexts[i] for i in range(len(contexts))],
        freq=[0] * len(contexts),
    )
    point_forecast = np.asarray(point_forecast, dtype=np.float32)[:, :horizon]

    out = ARTIFACTS / args.out
    np.savez(out, point_forecast=point_forecast)
    meta = {
        "runtime": "pytorch",
        "backend_class": backend_used,
        "repo": args.repo,
        "shape": list(point_forecast.shape),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}  shape={point_forecast.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
