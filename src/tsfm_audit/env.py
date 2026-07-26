"""Minimal .env loading, so API tokens stay out of shell history and out of git.

Deliberately dependency-free: this reads a handful of ``KEY=value`` lines and
nothing more. Real environment variables always win over the file, so CI and
Docker can override without editing anything.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import REPO_ROOT


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Ignores blanks, comments, and ``export`` prefixes."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_dotenv(path: Path | None = None) -> list[str]:
    """Load ``.env`` into the environment without overriding what is already set.

    Returns the names of the keys it set (never the values — these are secrets).
    """
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return []
    applied = []
    for key, value in parse_dotenv(path.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    return applied
