# RecRAG – local development helpers
#
# Usage:
#   make install    – one-time setup (uv sync + pnpm install)
#   make dev        – start all three services concurrently
#   make retrieval  – retrieval API only  (port 8000)
#   make ingestion  – ingestion API only  (port 8001)
#   make frontend   – Next.js dev server  (port 3000)
#   make ingest     – one-shot PDF ingestion (--force)
#   make watch      – file-watcher daemon
#   make lint       – ruff check
#   make format     – ruff format
#   make test       – pytest
#   make typecheck  – mypy
#   make help       – this message

.PHONY: install dev retrieval ingestion frontend ingest watch \
        lint format test typecheck help

# PYTHONPATH is set inline here so .env only needs to carry API keys.
# --env-file .env is still passed so OPENAI_API_KEY etc. are loaded.
PYTHON_ENV := PYTHONPATH=src uv run --env-file .env

# ── One-time setup ─────────────────────────────────────────────────────────────

install:
	uv sync
	uv pip install -e .
	cd frontend && pnpm install

# ── Run all services ───────────────────────────────────────────────────────────
# Each process gets its own log prefix. Ctrl-C kills all three.

dev:
	@echo "Starting retrieval API  → http://localhost:8000"
	@echo "Starting ingestion API  → http://localhost:8001"
	@echo "Starting Next.js        → http://localhost:3000"
	@echo "Press Ctrl-C to stop all services."
	@$(PYTHON_ENV) uvicorn api.retrieval.main:app \
	    --host 0.0.0.0 --port 8000 --reload \
	    2>&1 | sed 's/^/[retrieval] /' &
	@$(PYTHON_ENV) uvicorn api.ingestion.main:app \
	    --host 0.0.0.0 --port 8001 --reload \
	    2>&1 | sed 's/^/[ingestion] /' &
	@cd frontend && pnpm dev 2>&1 | sed 's/^/[frontend]  /' &
	@wait

# ── Individual services ────────────────────────────────────────────────────────

retrieval:
	$(PYTHON_ENV) uvicorn api.retrieval.main:app \
	    --host 0.0.0.0 --port 8000 --reload

ingestion:
	$(PYTHON_ENV) uvicorn api.ingestion.main:app \
	    --host 0.0.0.0 --port 8001 --reload

frontend:
	cd frontend && pnpm dev

# ── Backend CLI tools ──────────────────────────────────────────────────────────

ingest:
	$(PYTHON_ENV) python cli/ingest.py --force

watch:
	$(PYTHON_ENV) python cli/watch.py

# ── Quality checks ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

typecheck:
	uv run mypy src/

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make install    – one-time setup (uv sync + pnpm install)"
	@echo "  make dev        – start retrieval, ingestion, and frontend together"
	@echo "  make retrieval  – retrieval API only  (http://localhost:8000)"
	@echo "  make ingestion  – ingestion API only  (http://localhost:8001)"
	@echo "  make frontend   – Next.js dev server  (http://localhost:3000)"
	@echo "  make ingest     – one-shot PDF ingestion"
	@echo "  make watch      – file-watcher daemon"
	@echo "  make lint       – ruff check"
	@echo "  make format     – ruff format"
	@echo "  make test       – pytest"
	@echo "  make typecheck  – mypy"
	@echo ""
