"""Run Lag-Llama on one GIFT-Eval task, in its own pinned environment.

A caveat that shapes this whole script: GIFT-Eval publishes a Lag-Llama score but
**not the configuration that produced it**. Its entry declares
``replication_code_available: No`` and, alone among the models, has no notebook.
Lag-Llama's context length is an explicit tunable - the model card recommends
trying 32, 64, 128, 256, 512 - and the score moves with it.

So a single run cannot validate this harness: a mismatch could be our code or
their unstated context length, and the two are indistinguishable. Instead we
sweep the model card's own recommended values and report the whole curve. If one
lands on the published number, that is evidence for both the setting and our
wiring; if none do, that is a reportable reproducibility failure rather than a
bug we can locate.

The candidate set is the model card's list, fixed before running, so this is a
stated sweep rather than a search for a flattering match.

    uv run python run_gift.py --task ett1/H/short
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

# From the Lag-Llama model card. Fixed before any run.
CANDIDATE_CONTEXTS = (32, 64, 128, 256, 512)
NUM_SAMPLES = 100


def load_checkpoint() -> Path:
    """Fetch the pinned Lag-Llama checkpoint."""
    import json

    from huggingface_hub import hf_hub_download

    lock = json.loads((REPO_ROOT / "model_revisions.lock.json").read_text(encoding="utf-8"))
    entry = lock["models"]["lag-llama"]
    path = hf_hub_download(
        repo_id=entry["repo_id"],
        revision=entry["revision"],
        filename="lag-llama.ckpt",
        local_dir=str(REPO_ROOT / "data" / "checkpoints" / "lag-llama"),
    )
    return Path(path)


def build_gluonts_dataset(windows: list[gift.GiftWindow], freq: str) -> list[dict]:
    """Lag-Llama uses calendar features, so each entry needs a real start stamp."""
    return [
        {
            "start": pd.Period(window.start, freq=freq),
            "target": np.asarray(window.past, dtype=np.float32),
            "item_id": f"s{window.series_index}_w{window.window_index}",
        }
        for window in windows
    ]


def score_context(
    ckpt_path: Path,
    windows: list[gift.GiftWindow],
    horizon: int,
    season: int,
    freq: str,
    context_length: int,
    batch_size: int,
    device: str,
    no_rope_scaling: bool = False,
    progress_every: int = 10,
) -> tuple[float, float]:
    from lag_llama.gluon.estimator import LagLlamaEstimator

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    kwargs = checkpoint["hyper_parameters"]["model_kwargs"]

    # RoPE scaling. The model was trained at context 32 (kwargs["context_length"]),
    # and its rotary position embeddings are only meaningful over that range. The
    # model card is explicit: "enable RoPE scaling for the model to work well with
    # context lengths larger than what it was trained on."
    #
    # Leaving it None does not error or truncate - the embeddings simply
    # extrapolate past their trained range and the model degrades silently. Our
    # first sweep did exactly that at contexts 64 through 512, which is why its
    # curve was erratic rather than smooth.
    #
    # Linear scaling with factor (context + horizon) / trained_context, following
    # the authors' own zero-shot demo.
    trained_context = kwargs["context_length"]
    scaling_factor = max(1.0, (context_length + horizon) / trained_context)
    rope_scaling = (
        None
        if (no_rope_scaling or scaling_factor <= 1.0)
        else {"type": "linear", "factor": scaling_factor}
    )

    estimator = LagLlamaEstimator(
        ckpt_path=str(ckpt_path),
        prediction_length=horizon,
        context_length=context_length,
        input_size=kwargs["input_size"],
        n_layer=kwargs["n_layer"],
        n_embd_per_head=kwargs["n_embd_per_head"],
        n_head=kwargs["n_head"],
        scaling=kwargs["scaling"],
        time_feat=kwargs["time_feat"],
        rope_scaling=rope_scaling,
        batch_size=batch_size,
        num_parallel_samples=NUM_SAMPLES,
        device=torch.device(device),
    )
    module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, module)

    dataset = build_gluonts_dataset(windows, freq)
    started = time.time()

    # Consumed window by window rather than materialised with list(). A silent
    # death at context 512 left no output and no file, and there was no way to
    # tell whether it failed at window 3 or window 138. Progress is printed as it
    # goes and partial scores are kept, so the next failure is diagnosable and a
    # late one still yields most of the data.
    scores: list[float] = []
    try:
        for i, (window, forecast) in enumerate(
            zip(windows, predictor.predict(dataset, num_samples=NUM_SAMPLES), strict=False), 1
        ):
            median = np.quantile(np.asarray(forecast.samples, dtype=float), 0.5, axis=0)
            scores.append(mase(window.target, median[:horizon], window.past, season))
            if i % progress_every == 0 or i == len(windows):
                elapsed = time.time() - started
                rate = elapsed / i
                print(
                    f"      {i}/{len(windows)} windows  {elapsed:.0f}s elapsed  "
                    f"{rate:.1f}s/window  eta {(len(windows) - i) * rate:.0f}s",
                    flush=True,
                )
    except Exception as exc:  # noqa: BLE001 - a partial result still localises the failure
        print(
            f"      FAILED after {len(scores)}/{len(windows)} windows: {type(exc).__name__}: {exc}",
            flush=True,
        )
        if not scores:
            raise

    if len(scores) < len(windows):
        print(
            f"      partial: scoring {len(scores)} of {len(windows)} windows - "
            "this MASE is NOT comparable to a complete run",
            flush=True,
        )
    return float(np.nanmean(scores)), time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ett1/H/short")
    parser.add_argument("--contexts", nargs="*", type=int, default=list(CANDIDATE_CONTEXTS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="lag_llama_context_sweep.csv")
    # Kept only so the broken first sweep stays reproducible for comparison.
    parser.add_argument("--no-rope-scaling", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    out = REPO_ROOT / "artifacts" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    windows, horizon = gift.load_task(args.task)
    freq = windows[0].freq
    season = get_seasonality(freq)
    ref = gift.published_mase("lag-llama", args.task)

    print(f"task {args.task}")
    print(f"  instances={len(windows)}  horizon={horizon}  seasonality={season}")
    print(f"  published MASE {ref:.4f}  (configuration NOT published)")
    print(f"  device={device}  sweeping contexts {args.contexts}\n")

    ckpt_path = load_checkpoint()
    torch.manual_seed(config.derive_seed("gift-smoke", "lag-llama", args.task))

    rows = []
    for context_length in args.contexts:
        try:
            our_mase, elapsed = score_context(
                ckpt_path,
                windows,
                horizon,
                season,
                freq,
                context_length,
                args.batch_size,
                device,
                no_rope_scaling=args.no_rope_scaling,
                progress_every=args.progress_every,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  context {context_length:<5} FAILED: {type(exc).__name__}: {exc}")
            continue
        dev = 100 * (our_mase - ref) / ref
        rows.append(
            {
                "task": args.task,
                "context_length": context_length,
                "MASE": our_mase,
                "ref_MASE": ref,
                "d_MASE_%": dev,
                "n_instances": len(windows),
                "rope_scaling": not args.no_rope_scaling,
                "device": device,
                "secs": round(elapsed, 1),
            }
        )
        print(
            f"  context {context_length:<5} MASE={our_mase:.4f}  "
            f"({dev:+.2f}% vs published)  {elapsed:.0f}s",
            flush=True,
        )
        # Written after every context, not at the end. Two GPU runs have been
        # killed mid-sweep by something outside this process, and a sweep that
        # dies on its third value should not discard the first two.
        existing = pd.read_csv(out) if out.exists() else None
        combined = (
            pd.DataFrame(rows)
            if existing is None
            else pd.concat(
                [existing[~existing["context_length"].isin([context_length])], pd.DataFrame(rows)],
                ignore_index=True,
            )
        )
        combined.drop_duplicates(subset=["task", "context_length"], keep="last").to_csv(
            out, index=False
        )

    if not rows:
        print("\nno context length produced a result")
        return 1

    frame = pd.read_csv(out)

    best = frame.iloc[frame["d_MASE_%"].abs().argmin()]
    print(
        f"\nclosest: context {int(best['context_length'])} at {best['d_MASE_%']:+.2f}% of published"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
