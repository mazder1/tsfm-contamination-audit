"""Resolve each audited model to a Hugging Face commit SHA and lock it.

Loading from ``main`` means a checkpoint can change under us between runs, which
would silently invalidate every result. This script resolves the current SHA for
each repo in the registry and writes ``model_revisions.lock.json``.

Run once, commit the lock file, and never load a model without it::

    uv run python scripts/pin_model_revisions.py

Repos that fail to resolve are reported and left unpinned. An unresolved repo id
is a real finding (wrong name, gated licence), not something to paper over.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsfm_audit import config  # noqa: E402

HF_API = "https://huggingface.co/api/models/{repo_id}"


def resolve(repo_id: str) -> dict:
    response = requests.get(
        HF_API.format(repo_id=repo_id),
        headers={"User-Agent": "tsfm-audit/0.1"},
        timeout=30,
    )
    if response.status_code == 404:
        return {"ok": False, "error": "404 not found"}
    if response.status_code in (401, 403):
        return {"ok": False, "error": "gated - accept the licence and set HF_TOKEN"}
    response.raise_for_status()
    payload = response.json()
    return {
        "ok": True,
        "sha": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
        "downloads": payload.get("downloads"),
    }


def main() -> int:
    entries: dict[str, dict] = {}
    failures: list[str] = []

    for model in config.AUDITED_MODELS:
        try:
            result = resolve(model.repo_id)
        except requests.RequestException as exc:
            result = {"ok": False, "error": str(exc)}

        if result.get("ok"):
            print(f"  ok      {model.key:<14} {model.repo_id}  ->  {result['sha']}")
            entries[model.key] = {
                "repo_id": model.repo_id,
                "revision": result["sha"],
                "last_modified": result.get("last_modified"),
            }
        else:
            print(f"  FAILED  {model.key:<14} {model.repo_id}  ({result['error']})")
            failures.append(model.key)
            entries[model.key] = {
                "repo_id": model.repo_id,
                "revision": None,
                "error": result["error"],
            }

    lock = {
        "schema_version": 1,
        "resolved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "models": entries,
    }
    config.MODEL_REVISION_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {config.MODEL_REVISION_LOCK.name}")

    if failures:
        print(f"unresolved: {', '.join(failures)} - fix the repo ids before Phase 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
