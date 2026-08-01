"""Phase 3.5 test 1, run properly: a forecaster with NO training corpus.

Seasonal naive cannot memorise anything - it has no weights and no training
data, it just repeats the last seasonal cycle. If IT scores better on real
series than on their surrogates, the gap is proven to be a property of the
surrogates, with no model memory anywhere in the loop.
"""

import numpy as np

from tsfm_audit import config
from tsfm_audit.analysis.metrics import get_seasonality, mase
from tsfm_audit.baselines.seasonal_naive import seasonal_naive_forecast
from tsfm_audit.series import load_series, split_at_gaps
from tsfm_audit.surrogates.block_bootstrap import block_bootstrap_ensemble, suggest_block_length
from tsfm_audit.surrogates.iaaft import iaaft_ensemble

HORIZON = config.EVAL_HORIZON
CONTEXT = config.EVAL_CONTEXT
N_SURR = 20


def naive_mase(values, season):
    past, target = values[:-HORIZON], values[-HORIZON:]
    return mase(target, seasonal_naive_forecast(past, HORIZON, season), past, season)


snapshot = sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
segments = [
    seg
    for s in load_series(snapshot)
    for seg in split_at_gaps(s, min_length=config.min_usable_segment_length())
]
chosen, seen = [], set()
for seg in segments:
    if seg.domain not in seen:
        chosen.append(seg)
        seen.add(seg.domain)

print(f"{'series':<46}{'real':>8}{'iaaft':>8}{'block':>8}{'gap_i%':>9}{'gap_b%':>9}")
for seg in chosen:
    values = np.asarray(seg.values, dtype=float)[-(CONTEXT + HORIZON) :]
    season = get_seasonality(seg.freq)
    block = suggest_block_length(values, season_length=season)

    real = naive_mase(values, season)
    iaaft_scores = [
        naive_mase(s, season)
        for s in iaaft_ensemble(
            values, [config.derive_seed("iaaft", seg.series_id, i) for i in range(N_SURR)]
        )
    ]
    block_scores = [
        naive_mase(s, season)
        for s in block_bootstrap_ensemble(
            values,
            [config.derive_seed("block", seg.series_id, i) for i in range(N_SURR)],
            block_length=block,
        )
    ]
    mi, mb = float(np.nanmedian(iaaft_scores)), float(np.nanmedian(block_scores))
    print(
        f"{seg.series_id:<46}{real:>8.3f}{mi:>8.3f}{mb:>8.3f}"
        f"{100 * (mi - real) / mi:>+9.1f}{100 * (mb - real) / mb:>+9.1f}"
    )
