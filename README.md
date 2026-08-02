# Contamination Audit of Time-Series Foundation Models

Time-series foundation models (Chronos, TimesFM, Moirai, Lag-Llama) claim *zero-shot*
forecasting: pretrained on large, partly undisclosed corpora, then evaluated on public
benchmarks they supposedly never saw. Those benchmarks are old, widely mirrored, and
plausibly inside the pretraining data. If they are, the headline "zero-shot beats
supervised" numbers are partly recall, not forecasting.

This repository measures that, and ships an evaluation set that cannot be contaminated by
construction.

**Status: mid-audit, findings live.** Pre-registration frozen; all four harnesses validated
against published numbers; the near-duplicate probe has produced verdicts on Moirai's
corpus; the behavioural probes are measured and their limits documented below. See
[`PLAN.md`](PLAN.md) for the full phase plan and every correction made along the way.

---

## The question, stated precisely

A model that saw a million electricity series will forecast an *unseen* electricity series
well. That is skill, and it is the whole point of pretraining. We are not measuring that.

> **Transfer is skill on the class. Recall is skill on the instance.**

We are asking whether these models perform better on *this specific series* than on a
freshly generated series with identical statistical properties. A surrogate is, by
construction, a new instance of the same class — so a systematic advantage on the real
series is instance-specific information the model should not have.

---

## Pre-registration

> Everything in this section was committed **before any model was run**. The git history is
> the evidence. Values are mirrored in [`src/tsfm_audit/config.py`](src/tsfm_audit/config.py)
> and enforced by [`tests/test_config.py`](tests/test_config.py), so drift between the
> stated protocol and the executed one fails CI.
>
> If any of this changes after results exist, the results are re-run, not the protocol
> retro-edited.

### Models under audit

Four checkpoints from four independent organisations, pinned by commit SHA in
[`model_revisions.lock.json`](model_revisions.lock.json). Nothing is ever loaded from `main`.

| Key | Repo | Org | Released |
|---|---|---|---|
| `chronos-base` | `amazon/chronos-t5-base` | Amazon | 2024-03-13 |
| `timesfm-200m` | `google/timesfm-1.0-200m` | Google | 2024-05-01 |
| `moirai-base` | `Salesforce/moirai-1.0-R-base` | Salesforce | 2024-03-01 |
| `lag-llama` | `time-series-foundation-models/Lag-Llama` | Morgan Stanley / Mila et al. | 2024-02-05 |

Moirai is included specifically because its training corpus (LOTSA) is public. It is the
one model where a near-duplicate search can be checked against ground truth.

### Metrics

- **Point:** MASE (scaled by the in-sample seasonal naive error).
- **Probabilistic:** CRPS.
- **Aggregation:** median of per-series ratios, reported **per dataset**.

Never a single global average. Averaging MASE across heterogeneous series hides exactly
the per-series structure the probe is looking for.

### Surrogates

`K = 100` surrogates per series, per family. Two families, and the pairing is the point:

| Family | Preserves | Destroys |
|---|---|---|
| **IAAFT** | Power spectrum, amplitude distribution | Nonlinear structure, instance identity |
| **Block bootstrap** | Local nonlinear dynamics | Long-range structure, instance identity |

| Result | Verdict |
|---|---|
| Gap under **both** families | Instance-specific advantage. **Memorization.** |
| Gap under **IAAFT only** | Model exploits nonlinearity. **Legitimate skill.** Not a finding. |
| No gap | Clean. |

A single-family design cannot separate these two, which is why the brief calls surrogate
design the decision the project rests on. Neither family is trusted until it passes the
surrogate null validation below. Surrogates are never stored — only their derived
seeds (`config.derive_seed`), which makes them regenerable and the storage cost zero.

### Statistical test

For series *i* under model *m*, with real score `s` (lower is better) and surrogate scores
`s_1..s_K`:

```
p_i    = (1 + #{k : s_k <= s}) / (K + 1)          # rank-based, one-sided
g_i    = (median(s_k) - s) / median(s_k)           # relative effect size
```

Multiplicity is controlled with **Benjamini-Hochberg FDR at q = 0.10**, applied across
series *within* each (model, dataset) cell. Reported intervals are bootstrap, 2000
resamples.

### Decision rule

Contamination is declared for a (model, dataset) cell when **all three** hold:

1. FDR-significant firing in **at least 20 %** of the cell's series, and
2. **under both surrogate families**, and
3. median effect size among firing series **exceeds the detection floor `δ`**.

**On `δ`.** The floor is not a number invented today — it is the output of a procedure
fixed today. In Phase 4 we fine-tune a small model on a corpus we control and measure a
dose-response curve. `δ` is then defined as:

> the smallest relative gap at which known-memorized series are detected with ≥ 80 % power
> at the same FDR level.

This is the honest resolution of a real tension: pre-registration demands a threshold now,
but a meaningful threshold requires calibration. Freezing the *rule* rather than the
*value* satisfies both. `config.PROTOCOL.detection_floor` is `None` until Phase 4 fills it,
and a test asserts that.

### Validity conditions (the run is void if these fail)

- **Surrogate null validation.** The surrogate argument assumes a legitimate forecaster
  relies only on the properties the surrogate preserves. That is an assumption about how a
  black box works, so it is tested rather than asserted. The measured gap must be
  statistically indistinguishable from zero, at the same FDR level, for:
  1. forecasters with no training corpus at all (seasonal naive, ETS, ARIMA), where
     memorization is impossible by construction;
  2. **each audited model run on the fresh benchmark**, where contamination is impossible
     by date — same weights, same probe, true gap known to be zero;
  3. pure noise.

  A model that fires under (2) relies on structure the surrogate destroys and **cannot be
  audited** until the surrogate is redesigned. Domain matching applies: fresh electricity
  load is the null control for the Electricity/ETT benchmarks, which is why the ENTSO-E
  source is load-bearing rather than optional.

- **Gap stability under harness perturbation.** The probe compares a model against itself, so
  systematic quirks in our pipeline should cancel in the subtraction - which is why our
  absolute scores need not match anyone's published number. That is an argument, not a
  measurement, so it is tested: the real-versus-surrogate gap must agree between harness
  configurations that shift the absolute level (batch size, dtype) to within the level shift
  they produce. A gap that moves with an arbitrary implementation choice is not measuring
  memorisation, and a probe can pass every silence test above while failing this one.

- **Negative control** fires on **< 5 %** of series. A probe that fires where contamination
  is impossible is broken, and every positive result it produced is worthless.
- **Positive control** (Phase 4's deliberately contaminated model) fires on series it was
  trained on.
- Pure-noise input produces no firing.
- A series duplicated inside a benchmark **does** fire.

### Gaps and segmentation

Real data has holes, and how they are patched can manufacture a finding. Filling a hole
invents easy-to-forecast data inside a series we are calling clean; gluing across one
invents a discontinuity. Either way the real series gains a feature its surrogates do not
have, the scores diverge, and the divergence looks exactly like memorization.

So nothing is filled and nothing is glued:

> Split every series at **every** gap — one hour or a hundred, no distinction. Then drop
> any segment shorter than one context window plus one forecast horizon, since it cannot
> produce a forecast at all.

The deliberate omission is a gap-length threshold. "Split only at gaps longer than *N*"
would be simpler, but *N* would be chosen with full knowledge of which windows it excluded.
The rule above has no such parameter — the only number in it comes from the models.

Minimum length is `EVAL_CONTEXT + EVAL_HORIZON` = **536**, one value for every model rather
than one each, so all four are scored on an identical set of segments and the null control
cannot change shape between them.

`EVAL_CONTEXT` is 512 because that is the largest context *every* audited model accepts:

| Model | Context | Evidence |
|---|---|---|
| Chronos-base | hard cap 512 | `config.json`: `n_positions`, `chronos_config.context_length` |
| TimesFM-200m | hard cap 512 | model card: "context lengths up to 512 time points" |
| Moirai-base | no cap | `max_seq_len` counts patches, not time steps |
| Lag-Llama | no cap | trained at 32; card recommends tuning 32–512 |

Above 512 the first two silently truncate; below it they are handicapped for nothing.
Per-model tuning is refused deliberately — Lag-Llama's card recommends tuning context per
dataset, which is the exact degree of freedom a contamination audit must not have. The cost
is Lag-Llama running far outside its training regime, reported rather than tuned away.

TimesFM additionally requires contiguous input with no holes, so segmentation is a hard
requirement of an audited model, not only our own hygiene.

### Reported power, not assumed power

A single context of 512 observations means very different things at different frequencies.
On hourly data it is three weeks, leaving ~13,000 forecast origins per series. On the daily
Wikipedia series it is most of the series: 566 observations minus 536 leaves **31** origins.

That is accepted rather than fixed. Shortening the context for daily data would restore the
power, but it would cost the property that makes the context defensible in the first place
— one value, chosen once, applied to everything, with no per-case dial.

The cost is real and worth stating plainly: web traffic is the *deliberately adversarial*
domain, included because TimesFM is documented as training on Wikipedia pageviews. So the
domain where we most want statistical power is the one that has least of it. Its intervals
will be wide, and a null result there is weak evidence rather than a clean acquittal.

The obligation this creates is that the disparity stays visible. `n_forecast_windows` is
recorded per series in every snapshot manifest, and web-traffic firing rates are never
presented as comparable in precision to the hourly domains.

On the current snapshot this affects one series: `entsoe:load:ES` splits into `#s1`
(2,819 h) and `#s2` (10,250 h), with a 449-hour tail after the July gap dropped as too
short. Segment numbering runs over all segments found, so a dropped one leaves a visible
hole rather than vanishing.

Implemented in [`series.split_at_gaps`](src/tsfm_audit/series.py) and covered by
[`tests/test_segments.py`](tests/test_segments.py).

### Fresh benchmark cutoff

Admissibility cutoff: **2025-01-01**.

The latest audited model was released 2024-05-01. Release date is an unreliable proxy for
data cutoff and most model cards are vague, so the cutoff carries a deliberate ~8 month
buffer. Every observation admitted post-dates every audited checkpoint by construction —
not because documentation says so, but because the data did not exist yet.

Enforced in `benchmark/fetch.py:enforce_admissibility`, which is the single load-bearing
function of the fresh benchmark and is covered by six dedicated tests.

---

## Fresh benchmark sources

Three domains, chosen so the benchmark is not a monoculture. All have deep archives, so the
window backfills rather than accruing.

| Source | Domain | Freq | Key |
|---|---|---|---|
| Open-Meteo archive | Weather (6 sites, both hemispheres) | Hourly | None |
| Wikimedia pageviews | Web traffic (6 articles) | Daily | None |
| ENTSO-E | Electricity load (3 bidding zones) | Hourly | `ENTSOE_API_TOKEN` |

Wikipedia pageviews is **deliberately adversarial**: TimesFM is documented as training on
this exact source. The domain is in-distribution for an audited model while the
observations are not — precisely the recall-vs-transfer separation we want to expose.

### ENTSO-E access

Unlike the other two, ENTSO-E is not open on request — API access is granted manually by
the Transparency Platform service desk. The route that worked, recorded so it is
reproducible:

1. Register at <https://transparency.entsoe.eu/>.
2. File a request with the service desk at <https://transparencyplatform.zendesk.com/>
   from the registered address, asking for RESTful API access and stating the account
   email and intended usage. (The older `transparency@entsoe.eu` mailbox is widely cited
   but the Zendesk desk is what answers.) Turnaround was one working day.
3. Once granted, generate a **Web Api Security Token** under *My Account*.
4. `cp .env.example .env` and paste the token into `ENTSOE_API_TOKEN`.

`.env` is gitignored and never overrides a real environment variable, so CI and Docker can
supply the token their own way. A missing token skips the source without failing the fetch.

Current snapshot (`fresh_20260727`): **15 series, 125,580 observations**, 2025-01-01 to
2026-07-20, all three sources live. Access was granted 2026-07-27 and the full window
backfilled in one run.

**Known gap.** `entsoe:load:ES` is missing 42 hours in two runs — 35 hours from
2025-04-28 11:00 UTC, coinciding with the Iberian Peninsula blackout, and 7 hours from
2026-07-01 00:00 UTC. PL and DE_LU are complete. Handled by the segmentation rule below,
fixed before any model was scored.

---

## Layout

```
src/tsfm_audit/
  config.py          pre-registered constants, seeds, model registry
  series.py          the one series container, with provenance
  benchmark/         fresh benchmark: sources + fetch + admissibility  [Phase 0 ✓]
  harness/           one interface over every audited checkpoint       [Phase 1]
  baselines/         seasonal naive, ETS, ARIMA, PatchTST, DLinear     [Phase 2]
  surrogates/        IAAFT, block bootstrap, validation suite          [Phase 3]
  probes/            surrogate gap, duplicate search, cutoff test      [Phases 5-7]
  analysis/          scoring, FDR, bootstrap intervals                 [Phase 5]
scripts/             pin_model_revisions.py
data/fresh/          snapshots + provenance manifests
```

## Reproducing

```bash
uv sync --extra dev --extra stats          # pinned via uv.lock
uv run pytest -m "not network"             # offline suite
uv run pytest -m network                   # live source smoke tests
uv run python -m tsfm_audit.benchmark.fetch
uv run python scripts/pin_model_revisions.py
```

Or in Docker: `docker build -t tsfm-audit . && docker run --rm tsfm-audit`.

CI runs lint, format check, and the offline suite on every push. It deliberately does not
hit third-party APIs or run a sweep — GitHub Actions has no GPU and a 6-hour cap. The real
sweep runs locally and its results are committed as artifacts.

---

## State of the probes

Three instruments were built. Their measured status, in one table:

| Probe | Status |
|---|---|
| **Near-duplicate corpus search** | **Works.** Gated on planted copies, semantics proven on real data. Sole source of hard verdicts. |
| **Surrogate gap** (IAAFT + block bootstrap) | **Retired.** Fails its null controls: provably clean forecasters show gaps of -133% to +74% on fresh data, unpredictable even in sign. Detailed post-mortem in PLAN.md Phase 3. |
| **Margin / generalization** | **Built, sensitivity ≈ zero.** On the one information-parity comparison available, the proven-contaminated model shows no advantage on its own training data — so this probe cannot convict, and fresh-side results are generalization evidence, never contamination clearance. |

## Findings

Each is documented with scripts, artifacts and its confounds in the commit history and PLAN.md.

**1. Ten benchmark datasets sit verbatim in Moirai's training corpus — test windows
included.** Proven by address (offset-level matches at RMS ≈ 0): m1_quarterly/yearly,
m4_yearly, m3_quarterly/yearly, fred_md, tourism_monthly/yearly, traffic,
australian_electricity. Declared overlap — the subset names match — but now verified at the
numbers level rather than assumed from labels.

**2. Dataset names are unreliable evidence of training overlap — in both directions.** Ten
other same-named LOTSA subsets contain *different data* than the benchmark of the same name
(covid_deaths, hospital, cif_2016, car_parts, both nn5, tourism_quarterly, m1_monthly,
m3_monthly, m4_quarterly). Name-based contamination claims — the field's usual currency,
including leaderboard leakage flags — are unsafe either way.

**3. ETT is not in Moirai's affordable corpus subsets.** 56 query windows against all 22
energy-domain subsets under a 120 MB cap: clean. Unsearched and named: three subsets of
0.7-1.8 GB and buildings_900k at 60 GB; resampled copies would also evade the method.

**4. Verbatim training-set presence shows no behavioural advantage — on the one fair test
we have.** Moirai on its own proven-contaminated traffic data: naive-ratio 0.68, versus 0.43
on fresh electricity under identical context/horizon/seasonality. Its short-series
contaminated sets are uninterpretable either way, because context length alone degrades the
model tenfold (measured on clean data). n = 1; scope stated.

**5. The models generalize.** On matched comparisons, Chronos and Moirai beat
contamination-immune baselines *more* on 2025-26 data that postdates their training than on
the old benchmarks (Chronos 0.28 fresh vs 0.82 old; Moirai 0.43 vs 0.68). The opposite of
what score-inflating memorisation would predict.

**6. Published numbers are widely unreproducible — baselines included.** Lag-Llama's
GIFT-Eval score depends on at least two unstated interacting settings (context, RoPE
scaling; details below). GIFT-Eval's AutoETS baseline was run non-seasonally — discovered by
config search, reproducing their 1.94 within 1% — which understates classical performance
and flatters every foundation model's margin over it. Their AutoARIMA and AutoTheta match
*no* configuration we tried, and ship no replication code. Meanwhile the same four classical
implementations reproduce the Chronos paper's Table 10 — which published its full procedure —
to within 1-5%. **One codebase matches every number whose procedure was published, and only
those.**

**7. The surrogate approach to contamination testing is unsound as commonly conceived.**
Statistically-matched fakes (spectrum, distribution) are not forecastability-matched: clean
linear models find them *easier* than real data, clean shape-based models find them harder,
and the sign flips per forecaster and domain. Any audit built on "model beats its
surrogates ⇒ memorisation" would manufacture findings. Acceptance battery for any redesign:
eight clean forecasters must read ≈ zero gap on fresh data.

## A finding from Phase 1

Three of the four audited models reproduce their published GIFT-Eval scores closely:
TimesFM +0.07%, Moirai -0.11%, Chronos -1.12%. The fourth cannot be checked at all.

Lag-Llama's benchmark entry publishes a score but not the configuration that produced it -
no replication notebook, and the entry says so itself. Running all five context lengths its
own model card recommends produces MASE from 0.9730 to 1.0152 against a published 0.9875,
with the two closest at opposite ends of the range and no monotonic trend.

Investigating that produced a defect on **our** side first: Lag-Llama trains at context 32
and its card requires RoPE scaling beyond that, which we had left at its default of `None`.
The model neither errors nor truncates in that state - it extrapolates its positional
embeddings and degrades silently. Correcting it moves results *further* from the published
figure, which suggests the benchmark also ran unscaled.

So there are at least two unstated settings behind that score, and they interact: one alone
swings the result by 2.6%. The number cannot be reproduced from published information - by
us or by anyone else. That is reported as a finding rather than filed as our limitation, and
it is a compact illustration of why this project exists: a widely-cited result no reader can
check.

The order matters and is recorded as it happened - the bug in our harness was found before
the conclusion about theirs was reached, not after.

Full detail in [`PLAN.md`](PLAN.md) under *Harness validation status*.

## Still open

- **The central question the evidence now poses:** does ingestion leave *any* behavioural
  fingerprint at all? Two paths remain: Phase 4's dose-response experiment (fine-tune a
  small model with known over-exposure; find where any probe starts firing) and the untried
  likelihood channel (does a model rate its training sequences as unusually probable even
  when forecasts don't improve).
- Lag-Llama and TimesFM fresh-side runs, and ARIMA on the fresh benchmark, to complete the
  generalization table.
- The four unsearched large corpus subsets, if the disk cost is ever accepted.
- Each model's true data cutoff — **Phase 7**, documenting established versus assumed.

## The result we are heading toward

When this project was pre-registered, the risk named here was finding nothing and being
tempted to loosen thresholds. The actual trajectory is stranger: the benchmarks are
unreproducible in five documented ways, the one inspectable corpus really does contain its
test data — and on every fair test so far, it doesn't help the models at all. If that holds
through the remaining work, the paper's claim becomes: *contamination of time-series
foundation models is real at the data level, invisible at the behavioural level, and the
field's evaluation hygiene is a bigger problem than its memorisation.*

Every correction, refuted prediction and retracted claim along the way is preserved in the
commit history — this project applies its standard to itself first.
