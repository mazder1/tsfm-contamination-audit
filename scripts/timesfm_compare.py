"""Compare TimesFM JAX and PyTorch forecasts on frozen inputs.

TimesFM 1.0 is deterministic - it emits quantile heads rather than sampling
trajectories - so this is an equality test with a tolerance for float32
arithmetic, not a statistical comparison. Any difference in MASE is systematic.

Thresholds are pre-registered in PLAN.md and fixed before either side was run.

    uv run python scripts/timesfm_compare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.analysis.metrics import get_seasonality, mase  # noqa: E402

# Pre-registered. See PLAN.md, "TimesFM checkpoint equivalence".
MAX_MEDIAN_RELATIVE_DIFF = 1e-3
MAX_MASE_DIFF_PERCENT = 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="timesfm_equivalence_inputs.npz")
    parser.add_argument("--jax", default="timesfm_forecasts_jax.npz")
    parser.add_argument("--pytorch", default="timesfm_forecasts_pytorch.npz")
    parser.add_argument("--seasonality-freq", default="H")
    args = parser.parse_args()

    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    frozen = np.load(artifacts / args.inputs)
    contexts, targets = frozen["contexts"], frozen["targets"]

    missing = [n for n in (args.jax, args.pytorch) if not (artifacts / n).exists()]
    if missing:
        print("missing forecast files: " + ", ".join(missing))
        print("run envs/timesfm/run_equivalence.py and the Linux container first")
        return 1

    jax_forecast = np.load(artifacts / args.jax)["point_forecast"]
    torch_forecast = np.load(artifacts / args.pytorch)["point_forecast"]

    if jax_forecast.shape != torch_forecast.shape:
        print(f"shape mismatch: jax {jax_forecast.shape} vs pytorch {torch_forecast.shape}")
        return 1

    absolute = np.abs(jax_forecast - torch_forecast)
    scale = np.maximum(np.abs(jax_forecast), 1e-8)
    relative = absolute / scale

    season = get_seasonality(args.seasonality_freq)
    jax_mase = float(
        np.nanmean(
            [mase(targets[i], jax_forecast[i], contexts[i], season) for i in range(len(targets))]
        )
    )
    torch_mase = float(
        np.nanmean(
            [mase(targets[i], torch_forecast[i], contexts[i], season) for i in range(len(targets))]
        )
    )
    mase_diff = 100 * abs(torch_mase - jax_mase) / jax_mase

    print(f"windows compared      {len(targets)}")
    print(f"forecast points       {jax_forecast.size}")
    print()
    print(
        f"median relative diff  {np.median(relative):.3e}   (limit {MAX_MEDIAN_RELATIVE_DIFF:.0e})"
    )
    print(f"p95 relative diff     {np.percentile(relative, 95):.3e}")
    print(f"max relative diff     {relative.max():.3e}")
    print(f"max absolute diff     {absolute.max():.6f}")
    print()
    print(f"MASE jax              {jax_mase:.6f}")
    print(f"MASE pytorch          {torch_mase:.6f}")
    print(f"MASE difference       {mase_diff:.4f}%   (limit {MAX_MASE_DIFF_PERCENT}%)")

    ok_rel = float(np.median(relative)) < MAX_MEDIAN_RELATIVE_DIFF
    ok_mase = mase_diff < MAX_MASE_DIFF_PERCENT
    print()
    print(f"relative-difference criterion  {'PASS' if ok_rel else 'FAIL'}")
    print(f"MASE criterion                 {'PASS' if ok_mase else 'FAIL'}")
    print()
    if ok_rel and ok_mase:
        print("EQUIVALENT - the PyTorch port may stand in for the JAX checkpoint.")
        return 0
    print("NOT EQUIVALENT - the substitution is not justified; report and reconsider.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
