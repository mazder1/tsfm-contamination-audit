# Per-model environments

The four audited models cannot share a Python environment. Each directory here is an
independently pinned environment plus a runner that emits the same CSV shape, so results
from different stacks can be compared without any of them being run outside the
configuration its authors published under.

The full reasoning, and the alternative that was rejected, is in
[`../PLAN.md`](../PLAN.md) under *One environment per model stack*.

| Directory | Model | Why it needs its own environment |
|---|---|---|
| `moirai/` | `Salesforce/moirai-1.0-R-base` | `uni2ts` requires `torch<2.5`; the main env is on 2.13. Also pins `einops==0.7.*` and `gluonts~=0.14.3` |
| `lag_llama/` | `time-series-foundation-models/Lag-Llama` | Not on PyPI; installs from GitHub. Lightning `.ckpt` checkpoint |
| `timesfm/` | `google/timesfm-1.0-200m` | JAX/PAX checkpoint rather than PyTorch |

Chronos needs no directory here - it runs in the main environment, which is where it was
validated.

## Running one

```bash
cd envs/moirai
uv sync
uv run python run_gift.py --task ett1/H/short
```

Each runner writes to `artifacts/gift_smoke.csv` in the repository root, appending a row
with the model, task, our MASE, the published MASE, and the deviation.

## If you are replicating

Each `uv.lock` in here is part of the result, not packaging detail. Forcing all four models
into a single environment resolves and produces numbers, but they are not comparable to
these - three of the four would be running outside their published configuration.
