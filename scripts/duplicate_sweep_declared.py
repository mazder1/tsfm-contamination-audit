"""Sweep all name-declared benchmark/LOTSA pairs through the verified matcher.

Same procedure that proved m1_quarterly: sample benchmark series, search every
corpus series, count verbatim matches. m5 is deferred (large download);
buildings_900k (60GB) is never pulled.

    uv run python scripts/duplicate_sweep_declared.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.probes.duplicates import find_matches_with_gaps  # noqa: E402

PAIRS = [
    ("monash_m1_monthly", "m1_monthly"),
    ("monash_m1_yearly", "m1_yearly"),
    ("m4_quarterly", "m4_quarterly"),
    ("m4_yearly", "m4_yearly"),
    ("monash_m3_monthly", "monash_m3_monthly"),
    ("monash_m3_quarterly", "monash_m3_quarterly"),
    ("monash_m3_yearly", "monash_m3_yearly"),
    ("monash_covid_deaths", "covid_deaths"),
    ("monash_fred_md", "fred_md"),
    ("monash_hospital", "hospital"),
    ("monash_cif_2016", "cif_2016_12"),
    ("nn5", "nn5_daily_with_missing"),
    ("monash_nn5_weekly", "nn5_weekly"),
    ("monash_tourism_monthly", "tourism_monthly"),
    ("monash_tourism_quarterly", "tourism_quarterly"),
    ("monash_tourism_yearly", "tourism_yearly"),
    ("monash_car_parts", "car_parts_with_missing"),
    ("monash_australian_electricity", "australian_electricity_demand"),
    ("monash_traffic", "traffic_hourly"),
    ("monash_weather", "weather"),
]
N_QUERIES = 15


def targets(row) -> list[np.ndarray]:
    t = np.asarray(row["target"], dtype=float)
    return [t] if t.ndim == 1 else [t[i] for i in range(t.shape[0])]


def main() -> int:
    import datasets

    rows = []
    for bench_name, lotsa_name in PAIRS:
        try:
            bench = datasets.load_dataset("autogluon/chronos_datasets", bench_name, split="train")
            bench.set_format("numpy")
            corpus = datasets.load_dataset("Salesforce/lotsa_data", lotsa_name, split="train")
            corpus.set_format("numpy")
            corpus_targets = [t for row in corpus for t in targets(row)]
        except Exception as exc:  # noqa: BLE001
            print(
                f"  {bench_name:<32} LOAD FAILED: {type(exc).__name__}: {str(exc)[:80]}", flush=True
            )
            rows.append({"benchmark": bench_name, "lotsa": lotsa_name, "error": str(exc)[:120]})
            continue

        step = max(1, len(bench) // N_QUERIES)
        indices = list(range(0, len(bench), step))[:N_QUERIES]
        matched = near = none = 0
        for i in indices:
            query = np.asarray(bench[i]["target"], dtype=float)
            query = query[~np.isnan(query)]
            best = None
            for target in corpus_targets:
                for m in find_matches_with_gaps(query, target):
                    if best is None or m.rms < best.rms:
                        best = m
            if best and best.kind == "match":
                matched += 1
            elif best:
                near += 1
            else:
                none += 1
        rows.append(
            {
                "benchmark": bench_name,
                "lotsa": lotsa_name,
                "n_corpus_series": len(corpus_targets),
                "queried": len(indices),
                "matched": matched,
                "near": near,
                "none": none,
            }
        )
        print(
            f"  {bench_name:<32} -> {lotsa_name:<32} "
            f"matched {matched}/{len(indices)}  near {near}  none {none}",
            flush=True,
        )

    out = Path(__file__).resolve().parents[1] / "artifacts" / "duplicate_sweep_declared.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
