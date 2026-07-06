# RecRAG – Local development and orchestration helpers
#
# Usage:
#   make help       – Show this message

.PHONY: install setup dev retrieval ingestion orchestrator frontend ingest evaluate \
        lint lint-backend lint-frontend format format-backend format-frontend \
        test typecheck check clean infra-up infra-down infra-logs \
        docker-up docker-down smoke-test help \
        version bump-major bump-minor bump-patch

# ── Variables ─────────────────────────────────────────────────────────────────

PYTHON_ENV := PYTHONPATH=src uv run --env-file .env
DOCKER_COMPOSE := docker compose
FRONTEND_DIR := frontend

# ── Setup ─────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## One-time setup: install Python and Node dependencies
	uv sync
	uv pip install -e .
	cd $(FRONTEND_DIR) && pnpm install
	@echo "Pre-caching LiteParse (first parse will be instant)..."
	@npx --yes @llamaindex/liteparse --version 2>/dev/null || true

setup: install ## Full setup: install dependencies and create .env
	@git config core.hooksPath .cursor/hooks
	@chmod +x .cursor/hooks/*.sh .cursor/hooks/pre-commit 2>/dev/null || true
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example. Please update it with your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# ── Development ───────────────────────────────────────────────────────────────

dev: ## Start retrieval, ingestion, orchestrator, and frontend together locally
	@echo "Starting retrieval API    → http://localhost:8000"
	@echo "Starting ingestion API    → http://localhost:8001"
	@echo "Starting orchestrator API → http://localhost:8002"
	@echo "Starting Next.js          → http://localhost:3000"
	@echo "Press Ctrl-C to stop all services."
	@trap 'kill 0' EXIT; \
	$(PYTHON_ENV) uvicorn app.retrieval.main:app \
	    --host 0.0.0.0 --port 8000 --reload \
	    2>&1 | sed 's/^/[retrieval] /' & \
	$(PYTHON_ENV) uvicorn app.ingestion.main:app \
	    --host 0.0.0.0 --port 8001 --reload \
	    2>&1 | sed 's/^/[ingestion] /' & \
	$(PYTHON_ENV) uvicorn app.orchestrator.main:app \
	    --host 0.0.0.0 --port 8002 --reload \
	    2>&1 | sed 's/^/[orchestrator] /' & \
	cd $(FRONTEND_DIR) && pnpm dev 2>&1 | sed 's/^/[frontend]  /' & \
	wait

retrieval: ## Start only the retrieval API
	$(PYTHON_ENV) uvicorn app.retrieval.main:app --host 0.0.0.0 --port 8000 --reload

ingestion: ## Start only the ingestion API
	$(PYTHON_ENV) uvicorn app.ingestion.main:app --host 0.0.0.0 --port 8001 --reload

orchestrator: ## Start only the orchestrator API
	$(PYTHON_ENV) uvicorn app.orchestrator.main:app --host 0.0.0.0 --port 8002 --reload

frontend: ## Start only the Next.js dev server
	cd $(FRONTEND_DIR) && pnpm dev

# ── Infrastructure ────────────────────────────────────────────────────────────

infra-up: ## Start optional local Milvus stack for development
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml up -d

infra-down: ## Stop optional local Milvus stack
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml down

infra-logs: ## Show logs for optional local Milvus stack
	$(DOCKER_COMPOSE) -f docker-compose.dev.yml logs -f

docker-up: ## Start the full stack (including apps) in Docker
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop the full stack
	$(DOCKER_COMPOSE) down

smoke-test: ## Run smoke tests against running services
	bash scripts/smoke-test.sh

# ── Data & Pipeline ───────────────────────────────────────────────────────────

ingest: ## Run one-shot PDF ingestion from jobs/ingest.py
	$(PYTHON_ENV) python jobs/ingest.py --force

evaluate: ## Run RAGAS evaluation on the dataset
	$(PYTHON_ENV) python jobs/evaluate.py

# ── Quality Checks ────────────────────────────────────────────────────────────

lint: lint-backend lint-frontend ## Run all linters

lint-backend: ## Run ruff check on backend
	uv run ruff check .

lint-frontend: ## Run biome check on frontend
	cd $(FRONTEND_DIR) && pnpm lint

format: format-backend format-frontend ## Run all formatters

format-backend: ## Run ruff format on backend
	uv run ruff format .

format-frontend: ## Run biome format on frontend
	cd $(FRONTEND_DIR) && pnpm format

test: ## Run backend tests with pytest
	uv run pytest

typecheck: ## Run mypy type checking
	uv run mypy src/

check: lint typecheck test ## Run all quality checks (lint, typecheck, test)

# ── Maintenance ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches, and temporary files
	rm -rf .venv/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf $(FRONTEND_DIR)/node_modules/
	rm -rf $(FRONTEND_DIR)/.next/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned all temporary files and caches."

# ── Versioning ────────────────────────────────────────────────────────────────

version: ## Show the current version
	@cat VERSION

bump-patch: ## Bump patch version (0.1.0 → 0.1.1)
	bash scripts/bump-version.sh patch

bump-minor: ## Bump minor version (0.1.0 → 0.2.0)
	bash scripts/bump-version.sh minor

bump-major: ## Bump major version (0.1.0 → 1.0.0)
	bash scripts/bump-version.sh major
