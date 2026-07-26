# Project Brief: Contamination Audit of Time-Series Foundation Models

## The problem

Time-series foundation models (Chronos, TimesFM, Moirai, Lag-Llama) claim zero-shot
forecasting: pretrained on large, partly undisclosed corpora, then evaluated on public
benchmarks they supposedly never saw. Those benchmarks are old, widely mirrored, and
plausibly inside the pretraining data. If they are, the headline "zero-shot beats
supervised" numbers are partly recall, not forecasting. Nobody has measured this. Build a
system that measures it, and ship a benchmark that cannot be contaminated by construction.

## What you must deliver

1. **Model harness**: load several pretrained models behind one interface and produce
   forecasts for an arbitrary series, with pinned checkpoint revisions.
2. **Baseline suite**: seasonal naive, ETS, ARIMA, and one modern supervised model. No
   score means anything without these next to it.
3. **Surrogate generator**: produce statistically matched fake series that preserve the
   properties a legitimate forecaster uses and destroy the identity a memorizing model
   would recall.
4. **Contamination probes**: at minimum the surrogate gap, near-duplicate search against
   public corpora, and a data-cutoff discontinuity test.
5. **Negative control**: a model or dataset where contamination is known to be impossible,
   run through the full pipeline. If a probe fires there, the probe is broken and the
   results are worthless.
6. **Fresh benchmark**: an evaluation set built entirely from data published after every
   audited model's release, with a reproducible fetch script.
7. **Results report**: per model, per dataset, with confidence intervals and an explicit
   verdict.
8. **CI**: GitHub Actions runs tests on every push, and on a schedule refetches the fresh
   benchmark and republishes the leaderboard.
9. **Public leaderboard**: a static page with the current verdicts.
10. **Public GitHub repo** with a README explaining the method and your decisions, and one
    command that reproduces every number.

## Technologies

**Required (must use):**
- PyTorch and Hugging Face for loading pretrained checkpoints
- At least three independently-trained foundation models
- Classical forecasting baselines (statsforecast or equivalent)
- Proper scoring rules: a scaled point metric and a probabilistic one (MASE and CRPS)
- Surrogate data methods from nonlinear time-series analysis (phase randomization, IAAFT)
- A significance test with correction for multiple comparisons
- Docker for a reproducible environment
- GitHub Actions for CI and the scheduled refresh
- Fully pinned dependencies, model revisions and seeds

**Your choice (decide and justify in the README):**
- Which models to audit
- Which benchmark datasets to probe
- Which live data sources feed the fresh benchmark
- Which surrogate family, and what it must preserve
- The near-duplicate detection method
- The statistical test and the contamination threshold
- Where the leaderboard is hosted

## Decisions you must make yourself

Do not skip these. This is the point of the exercise.

- **Surrogate design.** The whole project rests here. Phase randomization preserves the
  power spectrum but destroys nonlinear structure, so a model doing better on the real
  series than the surrogate might be exploiting nonlinearity rather than remembering.
  Decide what a surrogate must preserve for the gap to mean memorization, and defend it.
- **Separating recall from transfer.** A model that saw a million electricity series will
  forecast an unseen electricity series well. That is skill. Define, operationally, what
  distinguishes it from having seen this exact series.
- **The metric.** Point or probabilistic, scaled how, aggregated across series how.
  Averaging MASE across heterogeneous series hides everything.
- **Significance.** How many surrogates per series, which test, and how you correct across
  hundreds of series without either drowning in false positives or losing all power.
- **The threshold.** What size of gap counts as contamination. Fix it before you look at
  results and say so in the README.
- **Duplicate matching.** Window length, normalization, and how close counts as a hit. Too
  loose and every daily seasonal series matches every other.
- **Establishing each model's cutoff.** Release date is not data cutoff, and most model
  cards are vague. Document what you could establish and what you had to assume.
- **Repo structure.**

## Working rules (so you actually learn)

- Build one probe at a time. A pipeline that runs three probes badly is worth less than
  one probe you trust.
- **Reproduce a published number before you attack it.** If you cannot match the paper's
  reported score on its own benchmark, stop and find out why. That is already a finding.
- Get the negative control passing before you report a single positive result.
- Never report a gap without a baseline beside it and an interval around it.
- Pre-register your threshold and your test. Write them in the README, commit it, then run
  the experiment.
- Break it on purpose: feed the pipeline pure noise, feed it a series you deliberately
  fine-tuned a model on, feed it a series duplicated inside the benchmark. Watch what the
  probes say.
- For every decision, put the options and tradeoffs on the table, then choose. Never
  accept a silent default.
- Write the report as you go, in your own words.

## Suggested milestones (order, not instructions)

1. Harness: one model, one dataset, reproduce the published score.
2. Baselines beside it. Confirm the published margin over naive is real.
3. Surrogate generator, plus a validation that surrogates actually match on the properties
   you claim.
4. Ground truth: fine-tune a small model on a corpus you control, so you know exactly what
   it memorized, and calibrate the probe's sensitivity against it.
5. Run the surrogate gap probe across models and datasets, with the negative control.
6. Near-duplicate search against the public corpora.
7. Fresh benchmark: fetch pipeline, first evaluation.
8. Cutoff discontinuity test.
9. Docker, CI, scheduled refresh, leaderboard.
10. Write the report.

## Definition of done

- A public leaderboard giving, per model and per dataset, a contamination verdict with an
  interval, and a clean score on the fresh benchmark.
- A fresh benchmark that refreshes itself on a schedule and stays uncontaminated by
  construction.
- Negative control passes and calibration against the known-contaminated model is
  documented.
- A report with real numbers, including any published result you could not reproduce.
- Public repo: README, tests, CI passing, one command reproduces everything.
- You can defend every statistical choice out loud, especially the surrogate design.

## Stretch (optional)

- Write it up for arXiv. This is a workshop paper if the numbers hold.
- Attribution: identify which public corpus a memorized series came from.
- Propose a contamination-safe evaluation protocol other people can adopt.
- Extend the method to other zero-shot claims, for example tabular foundation models.
- Train a small foundation model yourself on a fully known corpus, to give the field a
  permanent calibration target.
