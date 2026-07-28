"""TimesFM JAX/PAX side of the JAX-vs-PyTorch equivalence test.

Runs inside the container built from the Dockerfile beside this file. Reads the
frozen contexts, forecasts them, writes the point forecasts. Deliberately does
nothing else, so the only difference from the PyTorch side is the runtime.

    python run_equivalence.py --artifacts /artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JAX_REPO = "google/timesfm-1.0-200m"

# TimesFM 1.0-200m architecture. Fixed by the checkpoint, not chosen by us.
NUM_LAYERS = 20
MODEL_DIMS = 1280
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
CONTEXT_LEN = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="/artifacts")
    parser.add_argument("--inputs", default="timesfm_equivalence_inputs.npz")
    parser.add_argument("--out", default="timesfm_forecasts_jax.npz")
    parser.add_argument("--repo", default=JAX_REPO)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args()

    import timesfm

    artifacts = Path(args.artifacts)
    frozen = np.load(artifacts / args.inputs)
    contexts = frozen["contexts"]
    horizon = int(frozen["horizon"])
    print(f"loaded {contexts.shape[0]} contexts of {contexts.shape[1]}, horizon {horizon}")

    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=args.backend,
            per_core_batch_size=args.batch_size,
            horizon_len=horizon,
            context_len=CONTEXT_LEN,
            num_layers=NUM_LAYERS,
            model_dims=MODEL_DIMS,
            input_patch_len=INPUT_PATCH_LEN,
            output_patch_len=OUTPUT_PATCH_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=args.repo),
    )
    print(f"backend: {args.backend}  repo: {args.repo}")

    point_forecast, _ = model.forecast(
        [contexts[i] for i in range(len(contexts))],
        freq=[0] * len(contexts),
    )
    point_forecast = np.asarray(point_forecast, dtype=np.float32)[:, :horizon]

    out = artifacts / args.out
    np.savez(out, point_forecast=point_forecast)
    meta = {
        "runtime": "jax-pax",
        "backend": args.backend,
        "repo": args.repo,
        "shape": list(point_forecast.shape),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}  shape={point_forecast.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
