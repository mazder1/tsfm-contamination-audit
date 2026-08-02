"""Final matcher gate: rediscover a DECLARED overlap between LOTSA and a benchmark.

LOTSA lists m1_quarterly by name; the Chronos zero-shot benchmark evaluates
monash_m1_quarterly. If these are the same data, the matcher must find it. Only
after this fires is the matcher allowed near anything undeclared.

    uv run python scripts/duplicate_search_verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.probes.duplicates import find_matches  # noqa: E402

N_QUERIES = None  # None = all benchmark series


def main() -> int:
    import datasets

    print("loading benchmark side: autogluon/chronos_datasets monash_m1_quarterly")
    bench = datasets.load_dataset(
        "autogluon/chronos_datasets", "monash_m1_quarterly", split="train"
    )
    bench.set_format("numpy")

    print("loading corpus side: Salesforce/lotsa_data m1_quarterly")
    corpus = datasets.load_dataset("Salesforce/lotsa_data", "m1_quarterly", split="train")
    corpus.set_format("numpy")
    corpus_targets = [np.asarray(row["target"], dtype=float) for row in corpus]
    print(f"  corpus series: {len(corpus_targets)}")

    n_queries = len(bench) if N_QUERIES is None else min(N_QUERIES, len(bench))
    matched = near = missed = 0
    for i in range(n_queries):
        query = np.asarray(bench[i]["target"], dtype=float)
        best = None
        for j, target in enumerate(corpus_targets):
            for m in find_matches(query, target):
                if best is None or m.rms < best[1]:
                    best = (j, m.rms, m.kind)
        if best and best[2] == "match":
            matched += 1
        elif best:
            near += 1
        else:
            missed += 1
        # Print only the interesting rows; 200 identical MATCH lines say nothing.
        if not (best and best[2] == "match"):
            label = "near" if best else "none"
            print(
                f"  query {i:>3} (n={len(query):>4}) -> {label}"
                + (f" rms={best[1]:.4f} corpus#{best[0]}" if best else "")
            )

    print(f"\nmatched {matched}/{n_queries}  near-miss {near}  none {missed}")
    print("GATE PASSED" if matched >= int(0.9 * n_queries) else "GATE FAILED")
    return 0 if matched >= int(0.9 * n_queries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
