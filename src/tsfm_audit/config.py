"""Central configuration: seeds, model registry, benchmark windows.

Everything here is pre-registered. Changing a value in this file after results
exist invalidates those results — re-run, do not retro-edit.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FRESH_DIR = DATA_DIR / "fresh"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_REVISION_LOCK = REPO_ROOT / "model_revisions.lock.json"

# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

# Single global seed. Per-series / per-surrogate seeds are derived from it
# deterministically so surrogates are regenerable and never need storing.
GLOBAL_SEED = 20260726


def derive_seed(*parts: str | int) -> int:
    """Deterministically derive a sub-seed from the global seed.

    Stable across processes and platforms (unlike ``hash()``), so a surrogate
    can be regenerated from its identifiers alone.
    """
    import hashlib

    key = "|".join(str(p) for p in (GLOBAL_SEED, *parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


# --------------------------------------------------------------------------
# Models under audit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A pretrained model under audit.

    ``revision`` is resolved to a commit SHA by ``scripts/pin_model_revisions.py``
    and stored in ``model_revisions.lock.json``. Never load from ``main``.
    """

    key: str
    repo_id: str
    org: str
    # Public release date. NOTE: release date is NOT the data cutoff — see
    # README "Establishing each model's cutoff". Used only as a lower bound.
    released: dt.date
    notes: str = ""


AUDITED_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="chronos-base",
        repo_id="amazon/chronos-t5-base",
        org="Amazon",
        released=dt.date(2024, 3, 13),
        notes="T5 over quantised value tokens.",
    ),
    ModelSpec(
        key="timesfm-200m",
        repo_id="google/timesfm-1.0-200m",
        org="Google",
        released=dt.date(2024, 5, 1),
        notes="Decoder-only patched. Trained heavily on Google Trends / Wiki pageviews.",
    ),
    ModelSpec(
        key="moirai-base",
        repo_id="Salesforce/moirai-1.0-R-base",
        org="Salesforce",
        released=dt.date(2024, 3, 1),
        notes="Masked encoder. Training corpus (LOTSA) is public — our ground-truth anchor.",
    ),
    ModelSpec(
        key="lag-llama",
        repo_id="time-series-foundation-models/Lag-Llama",
        org="Morgan Stanley / Mila et al.",
        released=dt.date(2024, 2, 5),
        notes="Small decoder-only with lag features.",
    ),
)


def latest_model_release() -> dt.date:
    """Latest release date across audited models."""
    return max(m.released for m in AUDITED_MODELS)


# --------------------------------------------------------------------------
# Fresh benchmark window
# --------------------------------------------------------------------------

# Data before this date is not admitted to the fresh benchmark.
#
# Rationale (pre-registered): the latest audited model was released 2024-05-01.
# Release date is an unreliable proxy for data cutoff, so we add a deliberate
# ~8 month buffer and start at 2025-01-01. Any observation after this date
# post-dates every audited checkpoint by construction.
FRESH_BENCHMARK_START = dt.date(2025, 1, 1)

# Several archive APIs lag real time by a few days; leave a margin so a fetch
# never produces partial trailing windows.
FRESH_BENCHMARK_LAG_DAYS = 7


def fresh_benchmark_end(today: dt.date | None = None) -> dt.date:
    """Latest admissible date for a fetch run."""
    today = today or dt.date.today()
    return today - dt.timedelta(days=FRESH_BENCHMARK_LAG_DAYS)


# --------------------------------------------------------------------------
# Pre-registered evaluation protocol (see README § Pre-registration)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    """Frozen analysis parameters. Fixed before any model was run."""

    # Surrogates per series, per family.
    n_surrogates: int = 100
    surrogate_families: tuple[str, ...] = ("iaaft", "block_bootstrap")

    # Scoring.
    point_metric: str = "MASE"
    probabilistic_metric: str = "CRPS"
    aggregation: str = "median_of_per_series_ratios"

    # Significance.
    fdr_q: float = 0.10
    # Fraction of series in a (model, dataset) cell that must fire.
    firing_rate_threshold: float = 0.20
    # Negative control must stay below this or the whole run is void.
    negative_control_max_firing_rate: float = 0.05
    # Bootstrap resamples for reported intervals.
    n_bootstrap: int = 2000

    # Effect-size floor. Filled in by Phase 4 calibration via the procedure
    # fixed in the README; None until that phase completes.
    detection_floor: float | None = None


PROTOCOL = Protocol()


# --------------------------------------------------------------------------
# Gap handling (pre-registered; see README § Gaps and segmentation)
# --------------------------------------------------------------------------

# Forecast horizon used throughout the audit.
EVAL_HORIZON = 24

# Largest context window across AUDITED_MODELS. Measured from the model configs
# in Phase 1; None until then.
#
# The *rule* is what is pre-registered, not the number — the same construction
# used for Protocol.detection_floor. Fixing the rule now and measuring the value
# later is what stops the threshold from being chosen to suit a result. Taking
# the maximum across models (rather than each model's own context) keeps every
# model scored on an identical set of segments, so the null control cannot
# quietly change shape between models.
MAX_AUDITED_CONTEXT: int | None = None


def min_usable_segment_length(max_context: int | None = None) -> int:
    """Shortest segment that can still be scored: one context plus one horizon.

    A segment shorter than this cannot produce even a single forecast, so it is
    dropped. This is the whole of the segmentation policy — there is deliberately
    no tunable gap-length threshold, because any such threshold would have been
    picked knowing which series it excluded.
    """
    ctx = MAX_AUDITED_CONTEXT if max_context is None else max_context
    if ctx is None:
        raise ValueError(
            "MAX_AUDITED_CONTEXT is unset — it is measured from the model configs in "
            "Phase 1. Pass max_context explicitly for a provisional figure."
        )
    return ctx + EVAL_HORIZON


# --------------------------------------------------------------------------
# Fresh benchmark sources
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    key: str
    domain: str
    freq: str
    requires_token: bool = False
    token_env: str | None = None
    params: dict = field(default_factory=dict)


FRESH_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(key="open_meteo", domain="weather", freq="hourly"),
    SourceSpec(key="wikipedia", domain="web_traffic", freq="daily"),
    SourceSpec(
        key="entsoe",
        domain="electricity",
        freq="hourly",
        requires_token=True,
        token_env="ENTSOE_API_TOKEN",
    ),
)
