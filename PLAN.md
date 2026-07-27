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

- **Hard:** our M4 MASE matches the published MASE. If it does not, stop and find out why.
- **Soft:** our ETTh1 MASE falls within the spread of independently published values. Out
  of range does not halt the project, because M4 passing localises the fault — but it must
  be chased down and written up before Phase 3, not waved through.

Every downstream number is worthless if the harness disagrees with the literature on the
literature's own turf. A confirmed failure to reproduce is itself a reportable finding.

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
