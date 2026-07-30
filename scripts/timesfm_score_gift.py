"""Score TimesFM's container-produced forecasts against the published GIFT-Eval number.

The validation the equivalence test cleared the way for, and the same check
Chronos (-1.12%) and Moirai (-0.11%) passed. TimesFM cannot run on this host at
all - timesfm requires jax[cuda12], which has no Windows wheels - so the
forecasts are produced in the container and scored here.

MASE denominators come from the frozen input file, where they were computed from
each window's full history rather than the 512-point context the model sees.
Using the truncated context would change what MASE means relative to how the
other models were scored.

    uv run python scripts/timesfm_score_gift.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.benchmark import gift  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="timesfm_gift_inputs.npz")
    parser.add_argument("--forecasts", default="timesfm_gift_forecasts_pytorch.npz")
    parser.add_argument("--task", default="ett1/H/short")
    args = parser.parse_args()

    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    frozen_path = artifacts / args.inputs
    forecast_path = artifacts / args.forecasts

    for path in (frozen_path, forecast_path):
        if not path.exists():
            print(f"missing {path.name}")
            print("freeze the inputs here, then run the container - see envs/timesfm_jax/README.md")
            return 1

    frozen = np.load(frozen_path)
    targets = frozen["targets"]
    denominators = frozen["seasonal_error"]
    forecasts = np.load(forecast_path)["point_forecast"]

    if forecasts.shape != targets.shape:
        print(f"shape mismatch: forecasts {forecasts.shape} vs targets {targets.shape}")
        return 1

    per_window = np.abs(targets - forecasts).mean(axis=1) / denominators
    our_mase = float(np.nanmean(per_window))

    ref = gift.published_mase("timesfm-200m", args.task)
    dev = 100 * (our_mase - ref) / ref

    meta_path = frozen_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"task           {args.task}")
    print(f"windows        {len(targets)}")
    print(f"horizon        {targets.shape[1]}")
    print(f"seasonality    {meta.get('seasonality', '?')}")
    print()
    print(f"our MASE       {our_mase:.4f}")
    print(f"published MASE {ref:.4f}")
    print(f"deviation      {dev:+.2f}%")
    print()
    print("For scale: Chronos landed at -1.12% and Moirai at -0.11% on this task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
