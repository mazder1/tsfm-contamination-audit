"""TimesFM JAX-vs-PyTorch equivalence test. Runs both runtimes in one container.

Both sides run here rather than one per machine, and that is forced rather than
chosen: timesfm's "torch" extra itself requires jax[cuda12], which has no Windows
wheels, so there is no host-side PyTorch environment to compare from. The result
is a better test anyway - one machine, one library, one Python, differing only in
the runtime.

Reads the frozen contexts, forecasts them through each runtime, writes the point
forecasts. Deliberately does nothing else: no benchmark loading, no metrics.

    python run_equivalence.py --artifacts /artifacts
    python run_equivalence.py --artifacts /artifacts --backend cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The audited JAX/PAX checkpoint, and Google's PyTorch port of the same model.
JAX_REPO = "google/timesfm-1.0-200m"
PYTORCH_REPO = "google/timesfm-1.0-200m-pytorch"

# TimesFM 1.0-200m architecture. Fixed by the checkpoint, not chosen by us.
NUM_LAYERS = 20
MODEL_DIMS = 1280
INPUT_PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
CONTEXT_LEN = 512


def hparams(timesfm, horizon: int, backend: str, batch_size: int):
    return timesfm.TimesFmHparams(
        backend=backend,
        per_core_batch_size=batch_size,
        horizon_len=horizon,
        context_len=CONTEXT_LEN,
        num_layers=NUM_LAYERS,
        model_dims=MODEL_DIMS,
        input_patch_len=INPUT_PATCH_LEN,
        output_patch_len=OUTPUT_PATCH_LEN,
    )


def runtime_class(runtime: str):
    """Resolve a runtime to its implementation class.

    timesfm 1.2.9's __init__ exposes exactly one class, named TimesFm, chosen by
    a try/except: TimesFmJax if jax imports, else TimesFmTorch. There is no
    top-level TimesFmTorch to reach, so both sides are imported from their own
    modules instead. Same package, same version - only the entry point differs.
    """
    if runtime == "jax":
        from timesfm.timesfm_jax import TimesFmJax

        return TimesFmJax
    from timesfm.timesfm_torch import TimesFmTorch

    return TimesFmTorch


def run_one(timesfm, runtime: str, contexts, horizon, backend, batch_size):
    """Forecast the frozen contexts through one runtime. Returns point forecasts."""
    repo = JAX_REPO if runtime == "jax" else PYTORCH_REPO
    model = runtime_class(runtime)(
        hparams=hparams(timesfm, horizon, backend, batch_size),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo),
    )
    point_forecast, _ = model.forecast(
        [contexts[i] for i in range(len(contexts))],
        freq=[0] * len(contexts),
    )
    return np.asarray(point_forecast, dtype=np.float32)[:, :horizon]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="/artifacts")
    parser.add_argument("--inputs", default="timesfm_equivalence_inputs.npz")
    parser.add_argument(
        "--runtimes",
        nargs="*",
        default=["jax", "pytorch"],
        choices=["jax", "pytorch"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    # Output naming, so a later run cannot silently overwrite the equivalence
    # artifacts that a passing result already rests on.
    parser.add_argument("--out-prefix", default="timesfm_forecasts")
    args = parser.parse_args()

    import timesfm

    artifacts = Path(args.artifacts)
    frozen = np.load(artifacts / args.inputs)
    contexts = frozen["contexts"]
    horizon = int(frozen["horizon"])
    print(f"loaded {contexts.shape[0]} contexts of {contexts.shape[1]}, horizon {horizon}")

    repos = {"jax": JAX_REPO, "pytorch": PYTORCH_REPO}
    for runtime in args.runtimes:
        print(f"\n=== {runtime} ({repos[runtime]}) ===")
        forecast = run_one(timesfm, runtime, contexts, horizon, args.backend, args.batch_size)
        out = artifacts / f"{args.out_prefix}_{runtime}.npz"
        np.savez(out, point_forecast=forecast)
        meta = {
            "runtime": runtime,
            "backend": args.backend,
            "repo": repos[runtime],
            "shape": list(forecast.shape),
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"wrote {out}  shape={forecast.shape}")

    print("\nboth files written; run scripts/timesfm_compare.py on the host")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
