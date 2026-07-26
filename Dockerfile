# Reproducible environment for the TSFM contamination audit.
# Build:  docker build -t tsfm-audit .
# Run:    docker run --rm -it tsfm-audit uv run pytest
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/

WORKDIR /app

# Dependency layer first so source edits don't invalidate the install cache.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra stats --extra dev --no-install-project

COPY . .
RUN uv sync --locked --extra stats --extra dev

CMD ["uv", "run", "pytest", "-m", "not network"]
