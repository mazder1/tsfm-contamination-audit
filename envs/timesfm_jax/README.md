# TimesFM: JAX vs PyTorch equivalence test

The audited checkpoint, `google/timesfm-1.0-200m`, is a JAX/PAX artifact. It cannot run on
the Windows host: `jaxlib` publishes no cp311 Windows wheels, and JAX has no CUDA support on
Windows regardless. Google also published `google/timesfm-1.0-200m-pytorch`, which does run
there - but a port is not automatically the same function, and substituting it without
evidence would put an unverified assumption underneath a quarter of the audit.

So the substitution is earned rather than assumed.

## Why this is a clean test

TimesFM 1.0 is **deterministic**. It emits quantile heads directly rather than sampling
trajectories the way Chronos does. So this is an equality check with a tolerance for float32
arithmetic, not a statistical comparison - and any difference in MASE is systematic rather
than noise.

## Design

The two runtimes never coexist. They exchange files:

1. **Host** freezes the inputs once, so both sides provably see identical data:
   ```
   uv run python scripts/timesfm_freeze_inputs.py
   ```
   Writes `artifacts/timesfm_equivalence_inputs.npz` plus a sidecar `.json` carrying a
   SHA-256 of the contexts.

2. **Container** runs *both* runtimes:
   ```
   docker build -t tsfm-timesfm-jax envs/timesfm_jax
   docker run --rm --gpus all -v "$PWD/artifacts:/artifacts" tsfm-timesfm-jax
   ```
   Writes `artifacts/timesfm_forecasts_jax.npz` and `timesfm_forecasts_pytorch.npz`.
   Without GPU passthrough, add `--backend cpu` after the image name - slower, same numbers,
   which is all this test needs.

3. **Host** compares:
   ```
   uv run python scripts/timesfm_compare.py
   ```

The container's job is deliberately small - load checkpoints, read an array, write two
arrays. No benchmark loading, no metrics. Anyone running step 2 needs no context beyond this
file.

### Why both runtimes ended up in the container

Originally the PyTorch side was meant to run on the Windows host. It cannot:
`timesfm[torch]==1.2.9` itself requires `jax[cuda12]`, which publishes no Windows wheels. So
there is no host-side PyTorch environment to compare from, and `envs/timesfm/` was removed
after being written.

This is a better test regardless. One machine, one library version, one Python, differing
only in which runtime executes - the alternative would have confounded a port difference
with a difference between two operating systems and two CUDA stacks.

It also means **TimesFM runs in this container for the whole audit**, not just for this
test, which is one of the costs already accepted under *One environment per model stack*.

## Round 2: validating the harness against a published number

Equivalence was the prerequisite, not the validation. TimesFM still has to reproduce a
published GIFT-Eval score the way Chronos (-1.12%) and Moirai (-0.11%) did. Same
artifact-exchange pattern, so the container stays dumb.

Inputs are already frozen and committed: `artifacts/timesfm_gift_inputs.npz`, all 140
windows of `ett1/H/short`, contexts hashed `135f0354...`.

```
docker run --rm --gpus all -v "$PWD/artifacts:/artifacts" tsfm-timesfm-jax \
    --inputs timesfm_gift_inputs.npz \
    --out-prefix timesfm_gift_forecasts \
    --runtimes pytorch \
    --backend cpu
```

Then on the host:

```
uv run python scripts/timesfm_score_gift.py
```

Notes:

- `--out-prefix` is required. Without it the run would overwrite
  `timesfm_forecasts_*.npz`, which a passing equivalence result rests on.
- `--runtimes pytorch` alone is enough now that the two are proven equivalent. Run both if
  you want the JAX number for the record; it costs another pass.
- 140 windows rather than 20, so expect roughly seven times the equivalence run's duration.
- The MASE denominators travel inside the frozen file, computed from each window's **full**
  history rather than the 512-point context the model sees. Chronos and Moirai were scored
  that way, and recomputing from the truncated context here would change what MASE means and
  quietly break the cross-model comparison.

### Why Python 3.10

`timesfm`'s PAX extra pins `paxml` and `lingvo` to `python_version == "3.10"` exactly. The
rest of this project is on 3.11, where those pins silently drop out and leave no PAX backend
to compare against.

## Pass criteria, fixed before either side was run

| Criterion | Limit |
|---|---|
| Median relative difference across all forecast points | < 1e-3 |
| Difference in resulting MASE | < 0.5% |

Both must hold. Picking these after seeing the numbers would be precisely the failure this
project exists to detect in other people's work.

**If they pass**, the PyTorch port stands in for the JAX checkpoint for the rest of the
audit, and the substitution is documented as measured rather than assumed.

**If they fail**, the substitution is not justified. TimesFM would then have to run through
JAX in a container for the whole audit, or be reported as unauditable on this hardware -
and the disagreement between two official releases of one model is itself worth reporting.

## Note on the timesfm version

Both sides pin `timesfm==1.2.9`. Comparing one library's two backends is a cleaner test than
comparing two library versions, which would confound a port difference with a version
difference.

## The configuration these results rest on

Both the equivalence proof and the GIFT-Eval score were produced by one image. Its resolved
versions are recorded in `artifacts/timesfm_runtime_versions.json` - read out of the image
with `importlib.metadata` rather than transcribed, and covering the full 249-package
environment as well as the ones that matter:

| | |
|---|---|
| python | 3.10.12 |
| timesfm | 1.2.9 |
| jax / jaxlib | 0.4.26 / 0.4.26 |
| torch | 2.13.0+cpu |
| paxml / praxis / lingvo | 1.4.0 / 1.4.0 / 0.12.7 |
| numpy | 1.26.4 |

Two of these are not incidental. `torch` is CPU-only and installed *outside* the `[torch]`
extra, because timesfm 1.2.9 marks that extra `python_version == "3.11"` while `[pax]`
requires exactly `3.10` - so `pip install "timesfm[pax,torch]"` resolves, exits 0, and
installs only one of the two runtimes. And both runtimes ran with `--backend cpu`, so
neither number depends on a GPU being present.

To regenerate after any change to the image:

```
docker run --rm --entrypoint python tsfm-timesfm-jax -c '...'  > artifacts/timesfm_runtime_versions.json
```

The exact command is in the commit that introduced the file. If the file and the image ever
disagree, the results in `artifacts/` belong to whichever configuration produced them - so
regenerate and re-run rather than editing the record.
