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
- **Start the fresh-benchmark fetcher on day one.** Non-obvious but important: the fresh
  benchmark needs data published *after* every audited model's release, so it has to
  accrue. Start it in month three and it's a three-week benchmark. Start it now and it
  collects quietly while everything else gets built. This is the only thing that is
  genuinely expensive to defer.

**Gate:** pre-registration committed before any model is run.

---

## Phase 1 — One model, one dataset, reproduce a published number

**Goal:** establish that the harness agrees with the literature before attacking it.

- Chronos-base on a dataset from its own paper. Nothing else. No surrogates, no probes.
- Expect to lose several days to preprocessing conventions — windowing, scaling,
  aggregation. That is normal, and it is the most valuable debugging in the project.

**Gate:** our MASE matches the published MASE. If it doesn't, stop and find out why.
Every downstream number is worthless if the harness disagrees with the paper on the
paper's own turf. A confirmed failure to reproduce is itself a reportable finding.

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
