"""Freeze the inputs for the TimesFM JAX-vs-PyTorch equivalence test.

Both runtimes must be fed provably identical data, and they never coexist on one
machine: JAX has no CUDA on Windows and no cp311 wheels there at all, so the JAX
side runs in a Linux container. Exchanging a frozen input file removes any
question of whether the two sides saw the same thing.

Contexts are truncated here rather than inside either runner, so no padding or
truncation convention can differ between them.

    uv run python scripts/timesfm_freeze_inputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.benchmark import gift  # noqa: E402

# TimesFM 1.0's architectural context cap.
CONTEXT_LEN = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ett1/H/short")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="timesfm_equivalence_inputs.npz")
    args = parser.parse_args()

    windows, horizon = gift.load_task(args.task)
    # Evenly spaced rather than the first N, so the sample spans series and
    # window positions instead of one series' early windows.
    step = max(1, len(windows) // args.n)
    chosen = windows[::step][: args.n]

    contexts = np.zeros((len(chosen), CONTEXT_LEN), dtype=np.float32)
    targets = np.zeros((len(chosen), horizon), dtype=np.float32)
    for i, window in enumerate(chosen):
        past = np.asarray(window.past, dtype=np.float32)[-CONTEXT_LEN:]
        if len(past) < CONTEXT_LEN:
            raise SystemExit(f"window {i} has only {len(past)} points; need {CONTEXT_LEN}")
        contexts[i] = past
        targets[i] = np.asarray(window.target, dtype=np.float32)[:horizon]

    out = Path(__file__).resolve().parents[1] / "artifacts" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, contexts=contexts, targets=targets, horizon=np.int64(horizon))

    digest = hashlib.sha256(contexts.tobytes()).hexdigest()
    meta = {
        "task": args.task,
        "n_windows": len(chosen),
        "context_len": CONTEXT_LEN,
        "horizon": int(horizon),
        "contexts_sha256": digest,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
