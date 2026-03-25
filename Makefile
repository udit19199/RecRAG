# RecRAG – Local development and orchestration helpers
#
# Usage:
#   make help       – Show this message

.PHONY: install setup dev retrieval ingestion frontend ingest evaluate \
        lint lint-backend lint-frontend format format-backend format-frontend \
        test typecheck check clean infra-up infra-down infra-logs \
        milvus-up milvus-down milvus-logs docker-up docker-down help

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

setup: install ## Full setup: install dependencies and create .env
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example. Please update it with your API keys."; \
	else \
		echo ".env file already exists."; \
	fi

# ── Development ───────────────────────────────────────────────────────────────

dev: ## Start retrieval, ingestion, and frontend together locally
	@echo "Starting retrieval API  → http://localhost:8000"
	@echo "Starting ingestion API  → http://localhost:8001"
	@echo "Starting Next.js        → http://localhost:3000"
	@echo "Press Ctrl-C to stop all services."
	@trap 'kill 0' EXIT; \
	$(PYTHON_ENV) uvicorn api.retrieval.main:app \
	    --host 0.0.0.0 --port 8000 --reload \
	    2>&1 | sed 's/^/[retrieval] /' & \
	$(PYTHON_ENV) uvicorn api.ingestion.main:app \
	    --host 0.0.0.0 --port 8001 --reload \
	    2>&1 | sed 's/^/[ingestion] /' & \
	cd $(FRONTEND_DIR) && pnpm dev 2>&1 | sed 's/^/[frontend]  /' & \
	wait

retrieval: ## Start only the retrieval API
	$(PYTHON_ENV) uvicorn api.retrieval.main:app --host 0.0.0.0 --port 8000 --reload

ingestion: ## Start only the ingestion API
	$(PYTHON_ENV) uvicorn api.ingestion.main:app --host 0.0.0.0 --port 8001 --reload

frontend: ## Start only the Next.js dev server
	cd $(FRONTEND_DIR) && pnpm dev

# ── Infrastructure ────────────────────────────────────────────────────────────

infra-up: ## Start infrastructure services (Milvus stack + Ollama)
	$(DOCKER_COMPOSE) --profile server up -d

infra-down: ## Stop infrastructure services
	$(DOCKER_COMPOSE) --profile server down

infra-logs: ## Show logs for infrastructure services
	$(DOCKER_COMPOSE) --profile server logs -f

milvus-up: ## Start only Milvus + etcd + MinIO
	$(DOCKER_COMPOSE) -f docker-compose.milvus.yml up -d

milvus-down: ## Stop only Milvus + etcd + MinIO
	$(DOCKER_COMPOSE) -f docker-compose.milvus.yml down

milvus-logs: ## Show logs for only Milvus + etcd + MinIO
	$(DOCKER_COMPOSE) -f docker-compose.milvus.yml logs -f

docker-up: ## Start the full stack (including apps) in Docker
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop the full stack
	$(DOCKER_COMPOSE) down

# ── Data & Pipeline ───────────────────────────────────────────────────────────

ingest: ## Run one-shot PDF ingestion from cli/ingest.py
	$(PYTHON_ENV) python cli/ingest.py --force

evaluate: ## Run RAGAS evaluation on the dataset
	$(PYTHON_ENV) python cli/evaluate.py

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
