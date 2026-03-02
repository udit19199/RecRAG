# RecRAG – single multi-stage Dockerfile
#
# Build targets:
#   base        – shared Python environment (deps installed, code copied)
#   api         – FastAPI services (retrieval :8000 or ingestion :8001)
#   worker      – file-watcher daemon (cli/watch.py)
#   evaluation  – batch RAGAS evaluation (cli/evaluate.py)

# ── base ──────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency manifests first (layer-cache friendly)
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev

# Copy application source
COPY config.toml ./
COPY src ./src
COPY api ./api
COPY cli ./cli

# src/ contains all importable library code
ENV PYTHONPATH=/app/src

# ── api ───────────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000 8001

# Default: retrieval API (override via docker-compose command:)
CMD ["uv", "run", "uvicorn", "api.retrieval.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── worker ────────────────────────────────────────────────────────────────────
FROM base AS worker

CMD ["uv", "run", "python", "cli/watch.py"]

# ── evaluation ────────────────────────────────────────────────────────────────
FROM base AS evaluation

CMD ["uv", "run", "python", "cli/evaluate.py"]
