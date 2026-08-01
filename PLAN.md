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
| Lag-Llama | 0.9730-1.0152 | 0.9875 | brackets it | ⚠ not identifiable — see below |

#### Lag-Llama: the configuration behind the published number cannot be recovered

GIFT-Eval published a Lag-Llama score but not the configuration that produced it. Its entry
declares `replication_code_available: No` and, alone among the four models, it ships no
notebook. Context length is an explicit tunable - the model card recommends trying 32, 64,
128, 256, 512 - and it moves the score, so no single run can separate our harness from their
unstated setting.

All five recommended values were run, against a published 0.9875:

| Context | Our MASE | Deviation |
|---|---|---|
| 32 | 0.9255 | -6.28% |
| 64 | 0.9515 | -3.64% |
| 128 | 1.0045 | **+1.73%** |
| 256 | 1.0152 | +2.80% |
| 512 | 0.9730 | **-1.47%** |

**The sweep does not identify their setting.** The two closest results sit at opposite ends
of the recommended range, and the curve is not monotonic - it climbs through 256 then falls
at 512.

#### A defect in our own harness, found before blaming theirs

That table was produced with a bug. Lag-Llama was trained at context **32**, a value the
checkpoint carries in `model_kwargs`, and its model card states plainly: *enable RoPE scaling
for the model to work well with context lengths larger than what it was trained on*.
`rope_scaling` defaults to `None`, and we never set it. Every row above 32 therefore ran the
model outside its trained positional range with no correction - 16x beyond it at context 512.

The failure mode is silent by design. The module raises no error and truncates nothing; the
rotary embeddings simply extrapolate and accuracy degrades without warning.

**Sampling noise is now excluded as the explanation, by measurement.** Two runs at context 64
with RoPE scaling differ only in batch size - 32 versus 256 - which changes nothing about the
method but does change the order in which random draws are consumed, making them two
independent samples of the model's sampling variance:

| Batch | MASE | Seconds |
|---|---|---|
| 32 | 0.9272 | 414 |
| 256 | 0.9294 | 1451 |

They differ by **0.24%**, against swings of up to **4.3%** across the context sweep. The
noise is roughly twenty times too small to account for the erratic curve, so that curve is
real model behaviour under positional extrapolation, not variance. The originally planned
noise-measurement exercise is therefore closed - and it would have measured the wrong thing.

Incidentally, the larger batch was 3.5x *slower*: 256 windows times 100 sampled trajectories
overwhelms an 8GB card, and torch here falls back to a memory-hungry attention path. Batch
tuning is not a route to shortening these runs.

**Correcting it does not rescue the published number**, but it matters a great deal for the
audit. At context 512 - the value Phase 5 will use, and 16x the trained context - scaled
gives 0.9305 against 0.9730 unscaled, a **4.4% swing**. Every Lag-Llama forecast in the real
sweep would have carried that error.

| Context | Unscaled | Scaled |
|---|---|---|
| 32 | 0.9255 | 0.9115 |
| 64 | 0.9515 | 0.9283 |
| 128 | 1.0045 | 0.9322 |
| 256 | 1.0152 | 0.9446 |
| 512 | 0.9730 | 0.9305 |

Scaling narrows the spread from 9.7% to 3.6% - about 2.7x tighter, not the order of
magnitude claimed here when only three scaled points existed.

**It does not make the curve monotonic.** Scaled 512 sits 1.5% below scaled 256, six times
the measured sampling noise of 0.24%, so that dip is real rather than variance. RoPE scaling
substantially reduces the erratic dependence on context length without eliminating it, and
why that remains is unexplained.

Both corrections came from running the middle contexts after the conclusion had already been
drawn from the endpoints. The first pass revised the band; the second revised the shape.

Scaling moves *further* from the published 0.9875 at every context tested, so GIFT-Eval
almost certainly ran with library defaults, unscaled, as we originally did.

**Batch size is a memory cliff, not a throughput dial.** The scaled 512 run first died
silently, then hung for an hour at 100% GPU utilisation with 157 MiB of 8192 free - Windows
spills VRAM to host memory rather than failing, so an out-of-memory condition presents as
extreme slowness. Dropping from batch 32 to batch 4 completed the same run in 21 minutes
against 2.7 hours, 7.8x faster from *less* parallelism. Torch here also falls back to a
memory-hungry attention path, since it was not compiled with flash attention.

**Available VRAM is not a constant, and the runs were never compared under equal conditions.**
Context 256 appeared pathological - far slower than the larger context 512 at the same batch
size - until the headroom was checked: 512 ran with 7269 MiB free, 256 with about 5000 MiB,
the difference held by browser and desktop applications. At batch 1 with the memory freed,
256 completed in 7 minutes at a steady 3.0 s/window. Nothing about the configuration was
unusual; the comparison was uncontrolled.

This is an operational lesson for Phase 5, which runs far more inference than this. Free
VRAM on this machine varies by gigabytes with what is open, and exceeding it degrades
silently rather than failing. A sweep runner should record free memory at startup, so a slow
overnight run can be diagnosed afterwards instead of being written off as bad luck.

#### What the finding actually is

There are **at least two** unstated settings behind that published score - context length and
RoPE scaling - and they interact. One of them alone swings the result by 2.6%. Recovering
their number would require knowing both, and neither was published.

So the finding survives, and is sharper than first written: the configuration space behind
that score is larger than a single dial, and none of it is public. A benchmark number that
no reader can reproduce from published information is worth reporting as a finding rather
than filed as our limitation - it is this project's premise in miniature.

Stated honestly, though: we reached that conclusion only after finding a real bug on our own
side first, and the earlier version of this section attributed the problem outward before
that check had been done.

**Outstanding.** One run at context 512 with RoPE scaling enabled - the configuration the
audit will actually use, and the one where the correction should matter most, since 512 is
16x the trained context against the 2x tested so far. It costs about 2.7 hours and there is
no cheap way to shorten it. It is not needed to validate anything against the published
number, which is settled as impossible; it is needed to know how the model behaves in the
configuration Phase 5 will run it in.

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

### Status: both families built; probe FAILS its null control ⚠

A forecastability pilot (run early, at review's insistence, rather than waiting for Phase
3.5) answered two questions on fresh 2025-26 data, where contamination is impossible by date
and the true gap is known to be zero:

1. **Are surrogates forecastable at all?** Yes. Every surrogate median lands in the same
   range as real series. The method is not hopeless.
2. **Does the probe stay silent where it must?** **No.** On fresh German electricity load,
   Chronos scored 33-37% better on the real series than on its surrogates, under *both*
   families - the exact pre-registered signature of memorisation, on data the model cannot
   have seen.

| Series (all fresh) | Real | IAAFT med. | Bootstrap med. | Reading |
|---|---|---|---|---|
| DE_LU electricity | 0.441 | 0.661 | 0.703 | False positive, both families |
| Nairobi temperature | 0.943 | 1.077 | 0.647 | Families disagree in sign |
| Wikipedia pageviews | 6.115 | 0.540 | 0.566 | Comparison meaningless |

**Diagnosis.** Both families destroy calendar structure. Real electricity load has weekday/
weekend rhythm carried in the values themselves; IAAFT scrambles calendar alignment and the
bootstrap glues Sundays onto Wednesdays. A real series is legitimately easier to forecast
than its surrogate, so the probe fires with no memorisation anywhere - on electricity, the
audit's flagship domain. The bootstrap additionally errs the *other* way on smooth data,
manufacturing artificially clean cycles that would mask real contamination. And on
spike-dominated web traffic, one spike inside the target window makes the whole comparison
noise.

**Consequence: the surrogate families must be redesigned before Phase 4** - calendar-aware
variants that preserve the weekly structure a legitimate forecaster uses (candidates:
permuting whole weeks; IAAFT within day-of-week strata). This is a revision to
pre-registered machinery and is recorded as such. The pilot then re-runs until the null
control reads zero; nothing downstream means anything until it does.

That this was caught by a 15-minute pilot on clean data - before a single audit number
existed - is the pre-registration discipline working as designed, on ourselves first.

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

**4. Gap stability under harness perturbation.** The other three tests ask whether the probe
fires where it should not. This one asks a different question: whether the probe's output is
stable under *our own* implementation choices.

The design rests on an argument that has been asserted here rather than measured. The probe
compares a model against itself - same weights, same harness, real series versus surrogate -
so the reasoning goes that any systematic quirk in our pipeline applies to both sides and
cancels in the subtraction. That is why our absolute scores need not match anyone's
published number for the gap to mean something.

It is a reasonable argument. It is not evidence, and this project does not get to rely on
unmeasured reasoning while accusing others of exactly that.

**The test.** Run the probe under two harness configurations that shift the absolute level
without changing the method - batch size, which alters the sampling RNG stream, and dtype,
float32 against bfloat16. Compare the *gaps*, not the levels. A natural perturbation is
already measured: batch size moved Lag-Llama's level by 0.24%.

**Criterion, fixed before running:** the real-versus-surrogate gap must agree between
configurations to within the level shift those configurations produce. If the level moves
0.24% and the gap moves 0.24% or less, cancellation holds. If the gap moves substantially
more than the level, it does not.

**If it fails, the probe design is wrong**, not merely imprecise, and that has to surface
before Phase 5 rather than after a sweep has produced findings that cannot be trusted.

**Credit:** added after review pointed out that cancellation was being assumed rather than
demonstrated - the same objection that produced this phase in the first place, applied to
the phase itself.

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
tests 1-3, at the same FDR level used for the real sweep, **and** must satisfy the stability
criterion in test 4. Any firing means the surrogate is broken; instability means the probe
is. Either way, fix it and re-run before Phase 4.

The two failure modes are independent. A probe can be silent where it should be silent and
still produce a gap that swings with an arbitrary implementation choice, and such a probe
would pass tests 1-3 while being worthless.

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
