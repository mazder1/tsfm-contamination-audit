"""Phase 3 gate: do the surrogates preserve what they claim, on real data?

Unit tests check the families against synthetic series. This runs them against
the fresh benchmark - real weather, web traffic and electricity load - because a
surrogate family that works on a clean sine wave and fails on real data would
pass the tests and fail the project.

What this does NOT establish is whether the preserved properties are sufficient
for forecasting. That is Phase 3.5, and it is the harder question.

    uv run python scripts/validate_surrogates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit import config  # noqa: E402
from tsfm_audit.analysis.metrics import get_seasonality  # noqa: E402
from tsfm_audit.series import load_series, split_at_gaps  # noqa: E402
from tsfm_audit.surrogates.block_bootstrap import (  # noqa: E402
    block_bootstrap_ensemble,
    suggest_block_length,
)
from tsfm_audit.surrogates.iaaft import iaaft, iaaft_ensemble  # noqa: E402
from tsfm_audit.surrogates.validation import validate  # noqa: E402

# Fewer than the pre-registered K=100 used for the probe itself: this measures
# ensemble-mean statistics, which stabilise quickly, and running 100 per series
# here would cost minutes for no extra confidence.
N_SURROGATES = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--max-points", type=int, default=4096)
    args = parser.parse_args()

    snapshot = (
        Path(args.snapshot)
        if args.snapshot
        else sorted(config.FRESH_DIR.glob("fresh_*.parquet"))[-1]
    )
    print(f"snapshot {snapshot.name}\n")

    series_list = load_series(snapshot)
    need = config.min_usable_segment_length()

    rows = []
    for series in series_list:
        for segment in split_at_gaps(series, min_length=need):
            values = np.asarray(segment.values, dtype=float)[-args.max_points :]
            season = get_seasonality(segment.freq)
            block = suggest_block_length(values, season_length=season)

            iaaft_seeds = [
                config.derive_seed("iaaft", segment.series_id, i) for i in range(N_SURROGATES)
            ]
            block_seeds = [
                config.derive_seed("block", segment.series_id, i) for i in range(N_SURROGATES)
            ]

            iaaft_report = validate(values, iaaft_ensemble(values, iaaft_seeds), family="iaaft")
            block_report = validate(
                values,
                block_bootstrap_ensemble(values, block_seeds, block_length=block),
                family="block_bootstrap",
            )
            single = iaaft(values, seed=iaaft_seeds[0])

            for report in (iaaft_report, block_report):
                rows.append(
                    {
                        "series_id": segment.series_id,
                        "domain": segment.domain,
                        "n": len(values),
                        "season": season,
                        "family": report.family,
                        "block_length": block if report.family == "block_bootstrap" else None,
                        "dist_max_rel_diff": report.distribution_max_rel_diff,
                        "acf_max_abs_diff": report.acf_max_abs_diff,
                        "spec_median_rel_diff": report.spectrum_median_rel_diff,
                        "iaaft_iterations": single.iterations if report.family == "iaaft" else None,
                        "iaaft_converged": single.converged if report.family == "iaaft" else None,
                    }
                )
            print(
                f"  {segment.series_id:<46} n={len(values):<6} "
                f"iaaft dist={iaaft_report.distribution_max_rel_diff:.2e} "
                f"acf={iaaft_report.acf_max_abs_diff:.3f} "
                f"spec={iaaft_report.spectrum_median_rel_diff:.3f} "
                f"({single.iterations} iters)",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "artifacts" / "surrogate_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    print("\n=== by family ===")
    summary = frame.groupby("family")[
        ["dist_max_rel_diff", "acf_max_abs_diff", "spec_median_rel_diff"]
    ].agg(["median", "max"])
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    iaaft_rows = frame[frame.family == "iaaft"]
    print(f"\nIAAFT converged on {int(iaaft_rows.iaaft_converged.sum())}/{len(iaaft_rows)} series")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
