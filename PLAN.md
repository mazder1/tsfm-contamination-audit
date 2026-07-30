# Audit Plan: Contamination Testing of Time-Series Foundation Models

> Working plan for the project described in `TSFM-AUDIT-PROJECT-BRIEF.md`.
> Status: **draft — under review.** Nothing built yet.

## Organizing principle

**You have to earn the right to make an accusation.**

Every phase before the sweep exists to make the sweep believable. The output of this
project is not "we found gaps" — it is "we built a probe whose sensitivity we measured,
and here is what it says." A gap without a calibrated detector behind it is an opinion.

Ordering rule: **cheap and decisive first, expensive and ambiguous last.**

---

## Phase 0 — Pre-register, then build the skeleton

**Goal:** make it impossible to fool ourselves later.

- Write the README *first*: models, datasets, metric, statistical test, and the gap size
  that counts as contamination. Commit it. The git timestamp is the proof the threshold
  wasn't tuned to the answer.
- Repo skeleton, pinned environment, pinned model revisions, pinned seeds.
- **Stand up the fresh-benchmark fetcher.** Sources are chosen for deep archives, so the
  window backfills in a single run rather than accruing in real time — an earlier draft of
  this plan claimed collection had to start immediately or the benchmark would be short,
  which is wrong for archive-backed sources and was the reason for choosing them. The
  *scheduled* refresh lands in Phase 8 with the leaderboard it feeds; deferring it costs
  nothing, because any gap can be backfilled later.

**Gate:** pre-registration committed before any model is run; one snapshot captured with a
provenance manifest.

**Status: complete.** Committed `efb1540` (public, CI green). Current snapshot
`fresh_20260727`: **15 series, 125,580 observations**, 2025-01-01 to 2026-07-20, from
Open-Meteo, Wikimedia pageviews, and ENTSO-E.

ENTSO-E API access was granted 2026-07-27 and the full electricity window backfilled in a
single run across all three bidding zones — which is what the archive-backed source design
was for, and is the concrete payoff of correcting the accrual claim in `de7e275`.

Grids verified complete: no duplicate timestamps, no missing rows, every series on a
regular step. One data-quality finding, in Phase 3.5 below.

---

## Phase 1 — One model, two datasets, reproduce a published number

**Goal:** establish that the harness agrees with the literature before attacking it.

Chronos-base only. No surrogates, no probes. Expect to lose several days to preprocessing
conventions — windowing, scaling, aggregation. That is normal, and it is the most valuable
debugging in the project.

### Two datasets, in this order

**1. M4 first.** The competition published exact per-method MASE, so the target is
unambiguous and the gate can stay hard. This tests whether our scoring arithmetic and model
invocation are correct.

**2. ETTh1 second.** Chronos reports zero-shot results as aggregated relative scores rather
than a clean per-dataset MASE, so the target has to come from independent papers, which
disagree among themselves. We get a range, not a number. This tests whether our hourly
windowing and scaling conventions match the field's — and it is the plumbing the rest of
the project actually reuses, including the Phase 3.5 electricity null control.

### Why both, rather than either

They fail differently, and that is the point. A single ambiguous test leaves us unable to
read its own failure:

| M4 | ETTh1 | Reading |
|---|---|---|
| pass | in range | Harness sound. Proceed. |
| pass | out of range | Core is correct; ETTh1 conventions are wrong. One place to look. |
| fail | — | Something basic is broken. Do not touch ETTh1 until it is fixed. |

With ETTh1 alone, an out-of-range result cannot distinguish "our harness is broken" from
"our conventions differ from those papers", and days go into guessing which. Running the
sharp test first makes the fuzzy one interpretable.

Ordering follows this plan's own rule: cheap and decisive first, expensive and ambiguous
last.

**Gate — two parts:**

- **Hard:** our seasonal-naive MASE matches the published MASE. Deterministic, so exact.
  If it does not match, stop and find out why. **Status: passed** — all 27 datasets,
  190,674 series, worst deviation 0.024%, WQL exact.
- **Soft:** our Chronos MASE falls inside the tolerance derived below. Chronos samples, so
  exact agreement is not available and a tolerance is unavoidable.

### The Chronos tolerance, derived rather than chosen

Three sources of disagreement, and only two can be measured:

| Source | Measured? | Finding |
|---|---|---|
| Sampling (20 draws per forecast) | Yes | Relative SD ≈ `18.5 / √n` percent |
| dtype (float32 vs the reference's bfloat16) | Yes | ≤ 0.15% effect; bfloat16 is 8.0× slower on CPU |
| Published figures average 3 training runs, 1 checkpoint released | **No** | Irreducible |

Seed noise scales as `1/√n`: measured relative SD was 6.54% at n=8, 1.61% at n=72, 0.78% at
n=203, giving `rel_SD · √n` of 18.5, 13.7, 11.2. The conservative constant is **18.5**. A
single fixed percentage band would therefore be wrong in both directions — far too tight on
`dominick` (n=100,014, 3σ = 0.18%) and far too loose on `ercot` (n=8, 3σ = 19.6%).

The third source is a **constant offset of unknown sign**. We measured our scores running
+1.0% to +1.6% above the published figures on all three probe datasets. The consistency
across datasets points at the checkpoint rather than at chance: the one revision Amazon
released is presumably a little below their three-run mean, and that offset then shows up
everywhere.

*Correction to an earlier draft of this section.* It claimed a single checkpoint must score
**worse** than an average of three runs, and made a negative deviation a hard failure. That
is wrong: averaging three *scores* gives the middle of the three, and one run is as likely
to land above it as below. The observed sign is a property of the released checkpoint, not
a law. Corrected before any full-benchmark Chronos number was produced.

One ambiguity survives and is left open rather than resolved conveniently: if the three runs
were combined by averaging their *forecasts* rather than their *scores*, that is an ensemble,
and an ensemble genuinely does beat a single model. The maintainers' wording does not say
which. Under that reading a positive offset would be expected; under the other, no sign is.

**Per-dataset pass:** relative deviation lies in

```
[ -2.5 - 3 x 18.5/√n ,  +2.5 + 3 x 18.5/√n ]   percent
```

Symmetric, because the sign is not predictable. The ±2.5% centre absorbs the checkpoint
offset; the `3 x 18.5/√n` term absorbs sampling noise.

**Aggregate pass:** median signed deviation across the 27 within ±2.5 percent.

**Hard fail:** a median offset larger than ±2.5%, in *either* direction, or per-dataset
deviations that are large where `n` is large. A checkpoint difference is a percent or two;
ten percent is a bug. What distinguishes them is magnitude, not sign.

**On the constant 18.5.** It is the run-to-run wobble of a *single* series, recovered from
`rel_SD · √n` on the three probes, which gave 18.5, 13.7 and 11.2. We take the largest,
which is the widest band and therefore the easiest test to pass. That is a choice that
flatters us, and it is recorded as one. The justification is that the deterministic
seasonal-naive gate already validated the scoring arithmetic, so this gate is the lesser
check, and a false failure would cost days chasing a bug that is not there. Five seeds is
also a thin basis for a standard deviation - the spread from 11.2 to 18.5 is consistent with
estimation noise alone.

**Exemption:** `monash_covid_deaths` (published MASE 46.9) is judged on absolute scale, not
percentage. Tiny denominators make relative deviation meaningless there.

**Seed allocation.** Noise is large exactly where series are few, so seeds are spent there:
5 seeds for datasets with n ≤ 500, averaged; 1 seed above that, where 3σ is already under
2%. Seeds derive from `config.GLOBAL_SEED` and are recorded with the results.

All of this is fixed before any Chronos number for the full benchmark has been seen. The
only Chronos figures observed so far are the three probe datasets used to measure the noise
itself, which is what a tolerance has to be measured from.

Every downstream number is worthless if the harness disagrees with the literature on the
literature's own turf. A confirmed failure to reproduce is itself a reportable finding.

### Harness validation status

Each model checked against a published number on GIFT-Eval `ett1/H/short`, since that is the
one benchmark where all four have published per-task results under a single protocol.

| Model | Ours | Published | Deviation | |
|---|---|---|---|---|
| TimesFM-200m | 0.9386 | 0.9380 | **+0.07%** | ✓ |
| Moirai-base | 0.8840 | 0.8850 | **-0.11%** | ✓ (vs 1.1-R-base, the scored checkpoint) |
| Chronos-base | 0.8306 | 0.8400 | **-1.12%** | ✓ (also exact on the Chronos benchmark) |
| Lag-Llama | see below | 0.9875 | — | ⚠ blocked |

**Lag-Llama cannot be validated this way.** GIFT-Eval published its score but not the
configuration behind it - its entry declares `replication_code_available: No` and, alone
among the four, it has no notebook. Context length is an explicit tunable that moves the
score, so a mismatch cannot distinguish our harness from their unstated setting. The
model-card sweep gives -6.28% at context 32 and -3.64% at 64, rising toward the published
figure; 128 and above are unrun. If none land on it, that is a reportable reproducibility
failure rather than a bug we can locate.

TimesFM's number comes from the PyTorch port, which was first shown equivalent to the
audited JAX checkpoint to seven significant figures - see *TimesFM checkpoint equivalence*.

### Context length: done, and it corrected the segmentation rule

Read from the pinned checkpoints rather than assumed:

| Model | Context | Evidence |
|---|---|---|
| Chronos-base | hard cap **512** | `config.json`: `n_positions` 512, `chronos_config.context_length` 512 |
| TimesFM-200m | hard cap **512** | model card: "context lengths up to 512 time points" |
| Moirai-base | **no cap** | `max_seq_len` 512 counts *patches*, not time steps (`patch_sizes` 8–128) |
| Lag-Llama | **no cap** | trained at 32; card recommends tuning across 32–512 |

This invalidated the original phrasing of the segmentation minimum, which took "the largest
context across audited models". Two models have an architectural ceiling and two accept
anything, so there is no maximum to take.

`config.EVAL_CONTEXT = 512` replaces it: a pre-registered choice, being the largest value
*every* model can accept. Above it, Chronos and TimesFM silently truncate; below it, both
are handicapped for nothing. One value for all four rather than one each — per-model tuning
would add a free parameter, and Lag-Llama's card explicitly recommends tuning context per
dataset, which is exactly the degree of freedom this audit must not have. The cost is that
Lag-Llama runs far outside its training regime; that is reported, not tuned away.

Incidentally, TimesFM's card requires contiguous input with no holes — so the Phase 3.5 gap
segmentation is a hard requirement of an audited model, not only our own hygiene.

---

## Phase 2 — Baselines beside it

**Goal:** no score means anything alone.

- Seasonal naive first — it is MASE's denominator anyway.
- ETS + ARIMA via statsforecast.
- One supervised neural model: PatchTST. Also DLinear, because it is a single linear
  layer and it is informative when it wins.

**Gate:** the published margin over seasonal naive reproduces.

**Cost note:** `auto_arima` across thousands of series is CPU-bound and slow. Expect the
classical baselines to cost more wall clock than the foundation models.

---

## Phase 3 — Surrogates, and a validation that they do what we claim

**Goal:** build a surrogate whose gap means *memorization* and not *skill*.

Primary family: **IAAFT** — preserves the power spectrum *and* the amplitude distribution
exactly, which is strictly stronger than plain phase randomization.

### The two-family design (the key decision)

IAAFT destroys nonlinear structure *and* instance identity at the same time, so a gap
under IAAFT alone is ambiguous — the model might be exploiting nonlinearity rather than
remembering. So use a **second family that preserves nonlinear dynamics** (block bootstrap
or twin surrogates) while still destroying instance identity. Then:

| Result | Interpretation |
|---|---|
| Gap under **both** families | Model recognizes *this specific series*. Memorization. |
| Gap under **IAAFT only** | Model is exploiting nonlinearity. Legitimate skill. Not a finding. |
| No gap | Clean. |

This is also the operational answer to *recall vs. transfer*:

> **Transfer is skill on the class. Recall is skill on the instance.**
> A surrogate is, by construction, a new instance of the same class.

### Validation suite

Prove surrogates match the real series on the properties we claim they preserve:
autocorrelation function, power spectrum, marginal distribution. If they don't match,
the probe is measuring the wrong thing.

**Gate:** validation suite passes for both families.

---

## Phase 3.5 — Null validation: does the surrogate preserve forecastability? ⭐

**Goal:** prove the probe stays silent where contamination is impossible, *before* trusting
it anywhere it might fire.

### Why this exists

The surrogate argument assumes a legitimate forecaster relies only on the properties the
surrogate preserves. That is an assumption about how the model works — and these are black
boxes. If a model legitimately exploits structure IAAFT happens to destroy, a gap appears
with no memorization behind it, and nothing downstream distinguishes that from a real
finding. The probe would be manufacturing its own results.

Phase 3 checks that surrogates *match on their claimed properties*. That is a different and
weaker question than whether those properties are **sufficient for forecasting**. This
phase asks the second one.

### Three tests, escalating

**1. Forecasters with no training corpus.** Seasonal naive, ETS, ARIMA — fit per-series,
so memorization is impossible by construction. Run the full probe on them.

A gap here means the surrogate destroyed structure a legitimate forecaster uses. Seasonal
naive is the sharpest diagnostic: simple enough that any gap points at a specific defect
(seasonality not preserved, IAAFT not converged).

**2. The audited models on the fresh benchmark.** The strong test, because it uses the
actual black box.

Same weights, same probe, but on post-2025 data that cannot be in any audited training set.
The true gap is known to be zero. If a gap appears anyway, that model relies on something
the surrogate destroys, and it **cannot be audited** until the surrogate is redesigned.

*Confound:* the fresh benchmark is different data, so a difference could be domain rather
than contamination. Mitigate by matching domains — fresh electricity load is the null
control for the Electricity/ETT benchmarks. This is why the ENTSO-E source is load-bearing
rather than optional.

#### Decided: the Iberian blackout in `entsoe:load:ES` ✓

The ES load series has 42 missing hours in exactly two runs:

| Window (UTC) | Length | Reading |
|---|---|---|
| 2025-04-28 11:00 → 2025-04-29 21:00 | 35 h | Coincides with the Iberian Peninsula blackout of 2025-04-28 |
| 2026-07-01 00:00 → 2026-07-01 06:00 | 7 h | Unexplained; short, and on a month boundary where reporting changes are common |

The first one matters. A 35-hour grid collapse is a genuine structural break sitting inside
the series that is supposed to serve as the **clean null control** for ETT and Electricity —
the two most-used benchmarks in the field. It is a problem for the null control specifically
because IAAFT preserves the amplitude distribution, so the outage is inherited by every
surrogate as scattered noise rather than as one contiguous event. Real and surrogate then
differ in a way that has nothing to do with memorization, which is precisely the false
positive Phase 3.5 exists to rule out.

**Decision: excise, and treat the series as segments.** No value is ever invented, and no
seam is ever glued. Taken before any model was run — the commit history is the evidence.

The rule, stated so that it contains no free parameter:

> Split every series at **every** gap, whether the gap is one hour or a hundred. Then drop
> any segment shorter than one context window plus one forecast horizon, because such a
> segment cannot produce a forecast at all.

The rejected alternative was "split only at gaps longer than *N* hours." It reaches the
same place for ES with less bookkeeping, but *N* would have been a number chosen while
already knowing which windows it excluded — a dial set to produce a preferred answer. The
rule above has no dial: the only threshold is dictated by the models' own context lengths.

Applied to the current snapshot, this touches exactly one series:

| Segment | Span | Hours | Kept |
|---|---|---|---|
| `entsoe:load:ES#s1` | 2025-01-01 → 2025-04-28 | 2,819 | yes |
| `entsoe:load:ES#s2` | 2025-04-29 → 2026-06-30 | 10,250 | yes |
| `entsoe:load:ES#s3` | 2026-07-01 → 2026-07-19 | 449 | no — too short |

The 7-hour gap still splits the series; the tail it leaves behind simply cannot be scored
and falls out on its own. No one had to decide that.

Segment ids are numbered over *all* segments found, not just the survivors, so a dropped
segment leaves a visible hole in the numbering rather than disappearing silently.

Implemented in `series.split_at_gaps`, with the minimum length from
`config.min_usable_segment_length` — currently **536** (context 512 + horizon 24).

**Correction to an earlier draft of this rule.** It originally set the minimum from "the
largest context across audited models", to be measured from the configs in Phase 1. Reading
the pinned checkpoints showed that quantity does not exist — see Phase 1 below. Context is
now a pre-registered choice of 512, which is the largest value every audited model can
accept. Corrected before any model was scored.

**3. Pure noise.** No structure to memorize, so nothing may fire.

#### Decided: unequal power across domains is accepted and reported ✓

A uniform context of 512 observations leaves the hourly series ~13,000 forecast origins and
the daily Wikipedia series **31**. Shortening the context for daily data would fix that, at
the price of the context no longer being one value applied to everything — a dial we
deliberately do not have.

So it is accepted. The consequence is that web traffic — the domain included *because*
TimesFM trained on Wikipedia pageviews, and therefore the sharpest recall-vs-transfer test
available — is also the domain with the least power behind it. A null result there is weak
evidence, not an acquittal, and must be reported as such.

`config.n_forecast_windows` is recorded per series in every manifest so the disparity is
visible in the data rather than only in prose. Decided before any model was scored.

**Gate — hard stop:** the measured gap must be statistically indistinguishable from zero in
all three, at the same FDR level used for the real sweep. Any firing means the surrogate is
broken; fix it and re-run before Phase 4.

**Credit:** this phase was added after review pointed out that the original design assumed
what the models use rather than testing it.

---

## Phase 4 — Calibration ⭐ (the keystone)

**Goal:** measure the probe's sensitivity against a model whose memory we control exactly.

Fine-tune a tiny model (Chronos-tiny, ~8M) on a corpus we fully control, deliberately
including some series and excluding others.

### One model, both controls

- On series it **saw** → positive control. The probe *must* fire.
- On series it **did not see** → negative control. The probe *must* stay silent.

### Dose-response curve

Vary exposure (epochs / repetitions of a series) and find where detection begins. This
yields a **detection floor**: "this probe detects memorization at ≥ N exposures."

That floor is what converts *"we found a gap"* into *"we found a gap larger than our
calibrated sensitivity"* — and it gives a **principled threshold** instead of an invented
one.

**Gate — hard stop:** do not run a single real model through the probe until the positive
control fires and the negative control stays silent. Everything after this phase is only
as trustworthy as this phase.

**Cost note:** the only phase involving real training. On an 8M model over a small
controlled corpus this is a couple of GPU-hours, not days.

---

## Phase 5 — The real sweep

**Goal:** run the calibrated probe across models and datasets.

- 4 models × selected benchmark datasets × both surrogate families.
- Per-series test comparing the real score against the surrogate distribution.
- Per-series p-value → **Benjamini–Hochberg FDR** correction across series.
- Report the **proportion of series firing per dataset**, with bootstrap intervals.

**Never report a single global average.** Averaging across heterogeneous series hides
everything. Never report a gap without a baseline beside it and an interval around it.

**Report window counts beside firing rates.** A 31-window daily series and a 13,000-window
hourly series do not carry the same weight, and a table that shows only firing rates makes
them look as if they do.

---

## Phase 6 — Near-duplicate search (scoped)

**Goal:** direct evidence, where the corpus is actually inspectable.

- Target **LOTSA** (Moirai's corpus) first — it is public and documented, which makes
  Moirai the one model where ground truth can be checked directly.
- Disk-hungry: hundreds of GB. Verify actual size before committing.
- Be explicit in the README about which corpora could **not** be searched.

---

## Phase 7 — Cutoff discontinuity test

**Goal:** third probe, lowest expected yield. Do it last.

The weakest of the three: model cards are vague, and release date ≠ data cutoff. Document
what could be established and what had to be assumed. Expect partly inconclusive results
and say so plainly.

---

## Phase 8 — Docker, CI, leaderboard, report

- CI runs the **pilot subset on synthetic data only** — GitHub Actions has no GPU and a
  6-hour cap. The real sweep runs locally; commit the results artifact.
- Scheduled job refetches the fresh benchmark and republishes the leaderboard.
- One command reproduces every number.

---

## Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| **Models** | Chronos-base, TimesFM-200m, Moirai-base, Lag-Llama | Four independent orgs, four architectures; Moirai's corpus is public, giving one ground-truth anchor |
| **Metric** | MASE (point) + CRPS (probabilistic) | Scaled and proper, as required |
| **Aggregation** | Median of per-series ratios, reported per-dataset | Mean-of-MASE across heterogeneous series is meaningless |
| **Surrogates** | IAAFT + a nonlinearity-preserving family | Disambiguates memorization from nonlinear skill |
| **Test** | Per-series rank test → Benjamini–Hochberg FDR | Controls false positives without destroying power |
| **Threshold** | Derived from the Phase 4 detection floor | Principled rather than invented; fixed before results are seen |
| **Environments** | One per model stack, independently pinned | Each model runs in the configuration its authors tested; see below |

### TimesFM checkpoint equivalence ⭐

The audited checkpoint `google/timesfm-1.0-200m` is a JAX/PAX artifact and cannot run on the
host: `jaxlib` ships no cp311 Windows wheels, and JAX has no CUDA support on Windows at all.
Google also published `google/timesfm-1.0-200m-pytorch`, which runs on the host GPU.

Using the port without evidence would place an unverified assumption under a quarter of the
audit - the exact failure mode this project accuses others of. So the substitution is earned.

**Why the test is clean.** TimesFM 1.0 is deterministic: it emits quantile heads rather than
sampling trajectories. Feeding both runtimes identical inputs is therefore an equality check
with a tolerance for float32 arithmetic, not a statistical comparison, and any MASE
difference is systematic rather than noise. This is a stronger position than was available
for Chronos.

**Design.** The runtimes never coexist. Inputs are frozen once on the host (with a SHA-256
of the contexts), the JAX side runs in a Linux container and returns a `.npz`, the PyTorch
side runs on the host, and a comparator reads both. The container's job is deliberately tiny
- load a checkpoint, read an array, write an array - so it can be executed by anyone without
this project's context. See `envs/timesfm_jax/README.md`.

**Pass criteria, fixed before either side was run:**

| Criterion | Limit |
|---|---|
| Median relative difference across all forecast points | < 1e-3 |
| Difference in resulting MASE | < 0.5% |

Both must hold. If they pass, the port stands in for the JAX checkpoint and the substitution
is recorded as measured. If they fail, the substitution is not justified: TimesFM runs
through JAX in a container for the whole audit, or is reported as unauditable on this
hardware. A disagreement between two official releases of one model would itself be worth
reporting.

Both sides pin `timesfm==1.2.9`, so the comparison is between one library's two backends
rather than between two library versions - which would confound a port difference with a
version difference.

### One environment per model stack ⭐

The four audited models cannot share a Python environment. `uni2ts`, which loads Moirai,
requires `torch<2.5`; Chronos is validated here against `torch 2.13`. `uni2ts` also pins
`einops==0.7.*` against Chronos's 0.8.2 and `gluonts~=0.14.3` against a current 0.16.3.
Lag-Llama is not on PyPI at all and installs from GitHub, and TimesFM 1.0 is a JAX/PAX
checkpoint rather than a PyTorch one.

**Rejected: one environment, everything downgraded to `torch 2.4`.** Simpler, one lockfile,
simpler Docker. Rejected anyway.

The decisive argument is not convenience but *what a shared environment would actually be
measuring*. Downgrading to satisfy the most restrictive dependency means running Chronos in
a configuration Amazon never tested, TimesFM in one Google never tested, and so on. Any
difference from a published number then has an extra candidate explanation - our dependency
resolution - that we could never rule out. This project's entire output is an accusation
about other people's numbers, so every avoidable source of doubt in our own has to go.

It would also have cost the Chronos reproduction we already have. Those numbers were
produced under `torch 2.13`; changing the environment invalidates them and forces a re-run.

**Costs accepted, explicitly:** four lockfiles instead of one, and a materially more complex
Docker story in Phase 8 - probably one image per stack rather than one image.

**Layout.** `envs/<model>/` holds an independently pinned environment plus a runner that
emits the same CSV shape as every other. The shared code they all import - `benchmark/gift.py`,
`analysis/metrics.py` - depends only on numpy, pandas and `datasets`, which every stack can
satisfy. Aggregation happens in the main environment.

**Note for anyone replicating.** It is tempting to force all four models into one
`torch 2.4` environment; it resolves, and it will produce numbers. Those numbers are not
comparable to these, because three of the four models would be running outside the
configuration they were published under. If you replicate, replicate the environments too -
each `envs/<model>/uv.lock` is part of the result, not packaging detail.

### Deliberate breakage (planned)

- Pure noise → nothing should fire.
- A series duplicated inside the benchmark → something should fire.
- The Phase 4 fine-tuned model on seen series → must fire.

---

## Compute budget

Not the constraint. Inference-only for the main audit; largest model is ~710M params.

- **Hardware:** one consumer GPU (12–24GB) is comfortable. 8GB or Apple Silicon works with
  fp16 / Bolt variants. CPU-only is viable with a reduced sweep.
- **Wall clock:** a full sweep is hours to a day. Development runs on a pilot subset.
- **Cost if renting:** likely under $100–200 total, including failed runs.
- **Disk:** ~50GB lean; 500GB+ if Phase 6 is done thoroughly.

### Cost-control rules

- **Store surrogate seeds, not surrogates.** Regenerate on demand. Makes reproducibility
  free, which the brief requires anyway.
- **Cache every forecast**, keyed by `(model revision, series hash, window, seed)`. The
  analysis gets re-run ~50 times; inference should be re-run ~twice.
- **Tier the sweep:** pilot (10×10, seconds) → dev (50×50, minutes) → full (overnight).
  Never debug on the full sweep.

---

## The risk worth naming

**We might find nothing.** All four models come back clean.

That is still a publishable result: *"we built a calibrated contamination probe, validated
its sensitivity against a known-contaminated model, and the headline zero-shot claims
survive"* is a real contribution.

The danger is that this outcome tempts a loosening of the threshold after the fact. Which
is exactly why Phase 0 comes first.
