"""The undeclared-copy hunt: is ETT hiding in LOTSA's energy subsets?

ETT does not appear in Moirai's corpus by name, and the declared sweep proved
names settle nothing. So: search the actual numbers of every ETT channel (ETTh
and ETTm, both stations) against every energy-domain subset small enough to pull.
Subsets over the size cap are skipped LOUDLY, never silently.

Queries: two 256-point windows per channel (start, middle), so a copy that
trimmed the series start is still found. Limitation recorded: a resampled copy
(e.g. 15-min data stored hourly) would not match and is not claimed excluded.

    uv run python scripts/duplicate_hunt_ett.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit.benchmark.published import _load_ett_frames  # noqa: E402
from tsfm_audit.probes.duplicates import find_matches_with_gaps  # noqa: E402

SIZE_CAP_MB = 120
CANDIDATES = [
    "australian_electricity_demand",
    "elecdemand",
    "elf",
    "spain",
    "gfc12_load",
    "gfc14_load",
    "gfc17_load",
    "covid19_energy",
    "residential_load_power",
    "residential_pv_power",
    "lcl",
    "london_smart_meters_with_missing",
    "bdg-2_bear",
    "bdg-2_fox",
    "bdg-2_panther",
    "bdg-2_rat",
    "smart",
    "ideal",
    "sceaux",
    "borealis",
    "bull",
    "hog",
    "cockatoo",
    "pdb",
    "solar_power",
    "wind_power",
]


def ett_queries() -> list[tuple[str, np.ndarray]]:
    out = []
    for name in ("ETTh", "ETTm"):
        for region, frame in enumerate(_load_ett_frames(name), 1):
            for col in frame.columns:
                if col == "timestamp":
                    continue
                v = frame[col].to_numpy(dtype=float)
                out.append((f"{name}{region}:{col}:start", v[:256]))
                out.append((f"{name}{region}:{col}:mid", v[len(v) // 2 : len(v) // 2 + 256]))
    return out


def main() -> int:
    import datasets
    from huggingface_hub import HfApi

    api = HfApi()
    queries = ett_queries()
    print(f"{len(queries)} ETT query windows", flush=True)

    rows = []
    for subset in CANDIDATES:
        try:
            files = api.list_repo_tree(
                "Salesforce/lotsa_data", path_in_repo=subset, repo_type="dataset"
            )
            mb = sum(getattr(f, "size", 0) or 0 for f in files) / 1e6
        except Exception as exc:  # noqa: BLE001
            print(f"  {subset:<36} SIZE CHECK FAILED {exc}", flush=True)
            continue
        if mb > SIZE_CAP_MB:
            print(f"  {subset:<36} SKIPPED ({mb:.0f} MB > cap)", flush=True)
            rows.append({"subset": subset, "mb": round(mb, 1), "status": "skipped_size"})
            continue
        try:
            corpus = datasets.load_dataset("Salesforce/lotsa_data", subset, split="train")
            corpus.set_format("numpy")
        except Exception as exc:  # noqa: BLE001
            print(f"  {subset:<36} LOAD FAILED {str(exc)[:60]}", flush=True)
            rows.append({"subset": subset, "mb": round(mb, 1), "status": "load_failed"})
            continue
        hits = []
        for row in corpus:
            t = np.asarray(row["target"], dtype=float)
            for series in [t] if t.ndim == 1 else [t[i] for i in range(t.shape[0])]:
                for qname, q in queries:
                    for m in find_matches_with_gaps(q, series):
                        if m.kind == "match":
                            hits.append((qname, row.get("item_id", "?"), m.rms))
        status = f"HIT x{len(hits)}" if hits else "clean"
        print(f"  {subset:<36} {mb:>7.1f} MB  {status}", flush=True)
        rows.append({"subset": subset, "mb": round(mb, 1), "status": status, "hits": hits[:20]})

    out = Path(__file__).resolve().parents[1] / "artifacts" / "duplicate_hunt_ett.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
