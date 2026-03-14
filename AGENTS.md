# AGENTS.md - Development Guidelines for RecRAG

Guidelines for agentic coding agents working in this repository.

## Project Overview

RecRAG is a Retrieval-Augmented Generation pipeline with separate ingestion and retrieval
microservices and a Next.js frontend for bulk PDF upload, indexing, and querying.

**Tech Stack:** Python 3.12+, FastAPI, pymilvus (Milvus/Zilliz), uv, Next.js, Docker,
OpenAI/Ollama/NVIDIA NIM

---

## Build / Lint / Test Commands

```bash
# ── One-time setup ────────────────────────────────────────────────────────────
make install          # uv sync + uv pip install -e . + pnpm install

# ── Local development (starts all three services) ─────────────────────────────
make dev              # retrieval :8000 | ingestion :8001 | frontend :3000

# ── Individual services ───────────────────────────────────────────────────────
make retrieval        # retrieval API only  (http://localhost:8000)
make ingestion        # ingestion API only  (http://localhost:8001)
make frontend         # Next.js dev server  (http://localhost:3000)

# ── Backend CLI tools ─────────────────────────────────────────────────────────
make ingest           # one-shot PDF ingestion (--force)

# ── Quality checks ────────────────────────────────────────────────────────────
make lint             # ruff check .
make format           # ruff format .
make test             # pytest
make typecheck        # mypy src/

# ── Pytest variants ───────────────────────────────────────────────────────────
uv run pytest tests/test_ingest.py                # single file
uv run pytest tests/test_ingest.py::test_name     # single test
uv run pytest -k "pattern"                        # filter by name

# ── Docker ────────────────────────────────────────────────────────────────────
docker-compose build          # build images
docker-compose up -d          # start all services
docker-compose down           # stop containers
```

> **`PYTHONPATH` note:** `PYTHONPATH=src` is set inline in every Makefile target so
> the `api` and `src` packages are importable by uvicorn/cli tools. `.env` only needs to
> contain API keys — **do not add `PYTHONPATH` back to `.env`**.
>
> If you run uvicorn directly (outside `make`), prefix the command:
> `PYTHONPATH=src uv run --env-file .env uvicorn api.retrieval.main:app ...`

---

## Code Style Guidelines

### Import Order
Group imports with blank lines between: 1) Standard library 2) Third-party 3) Local application

```python
import json
from pathlib import Path
from typing import Any, Optional

import requests
from openai import OpenAI

from adapters.base import BaseEmbedder
from config import load_config, resolve_path
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Functions/variables | snake_case | `get_documents()`, `file_path` |
| Classes | PascalCase | `DocumentLoader` |
| Constants | UPPER_SNAKE_CASE | `MAX_CHUNK_SIZE` |
| Private members | underscore prefix | `_internal_method()` |

### Type Hints
- Always use type hints for function parameters and return types
- Use `list[X]`, `dict[str, X]` instead of `List`, `Dict`
- Use `X | None` for optional values

```python
def load_config(config_path: Path = Path("config.toml")) -> dict[str, Any]:
def embed(self, text: str) -> list[float]:
```

### File Paths
- Use `pathlib.Path` instead of string paths
- Resolve paths relative to config file using `resolve_path()`

```python
from config import resolve_path
storage_dir = resolve_path(config["storage"]["directory"], config_path)
```

### Error Handling
- Use specific exception types, not bare `Exception`
- Provide meaningful error messages

### Docstrings
Write docstrings for all public functions and classes with Args and Returns sections.

---

## Key Patterns

### Adapter Pattern (`src/adapters/`)
- `BaseEmbedder` and `BaseLLM` are abstract base classes
- Factory functions in `__init__.py`: `create_embedder()`, `create_llm()`
- Each concrete adapter has a `provider` class attribute (`"openai"`, `"ollama"`, `"nim"`)

### Configuration (`src/config.py`)
- `load_config()`: Loads TOML with `${VAR:-default}` substitution
- `resolve_path()`: Resolves paths relative to config file
- `get_config_value()`: Gets nested config via dot notation

### Ollama API
- **Always set `"stream": False`** in request payloads
- Embeddings: `/api/embeddings` endpoint
- Generation: `/api/generate` endpoint

### Docker Networking
- Use `http://ollama:11434` (not `host.docker.internal`) for container-to-container

---

## Project Structure

```
RecRAG/
├── Makefile                  # Dev shortcuts (make dev, make install, etc.)
├── api/
│   ├── ingestion/
│   │   └── main.py           # FastAPI :8001 — upload, status, config, reindex
│   └── retrieval/
│       └── main.py           # FastAPI :8000 — query, health, config, providers
├── src/
│   ├── config.py             # Config loading & path resolution
│   ├── adapters/             # LLM & embedding providers (openai, ollama, nim)
│   ├── pipelines/            # Ingestion & retrieval pipeline logic
│   ├── stores/               # Vector store (Milvus)
│   ├── loaders/              # PDF document loader
│   ├── splitters/            # Text chunking
│   └── evaluation/           # RAGAS evaluation helpers
├── cli/
│   ├── ingest.py             # CLI: one-shot ingestion
│   └── evaluate.py           # CLI: batch RAGAS evaluation
├── frontend/                 # Next.js App Router
│   ├── app/
│   │   ├── page.tsx          # "/" — chat interface with sidebar document upload
│   │   └── ingest/page.tsx   # "/ingest" — standalone document management page
│   └── src/
│       ├── components/       # FileUploader, IngestionStatusDisplay, ModelPicker, Navbar
│       └── lib/api.ts        # Typed API client for both backend services
├── tests/                    # Pytest suite
├── Dockerfile                # Python image (APIs / evaluation)
├── Dockerfile.api            # Python image (FastAPI services)
├── docker-compose.yml        # Multi-container orchestration
├── config.toml               # Application config (non-sensitive)
├── .env                      # API keys only (not in git)
├── .env.example              # Template for .env
├── data/pdfs/                # PDF uploads (not in git)
└── storage/                  # Milvus state & ingestion status files (not in git)
```

---

## Configuration

### File Organization
- **`config.toml`**: All configuration settings (provider, model, URLs, chunk sizes, etc.)
- **`.env`**: API keys only (sensitive information) — no `PYTHONPATH` here

### Environment Variables (`.env`)
Only sensitive values:
- `OPENAI_API_KEY`: Required for OpenAI provider
- `NVIDIA_API_KEY`: Required for NVIDIA NIM provider (starts with "nvapi-")
- `OLLAMA_API_KEY`: Required if using Ollama Cloud API
- `MILVUS_HOST` / `MILVUS_USERNAME` / `MILVUS_PASSWORD`: Required for Zilliz Cloud

### Config Settings (`config.toml`)
All non-sensitive configuration including providers, models, and URLs:
```toml
[embedding]
provider = "openai"  # or "ollama", "nim"
model = "text-embedding-3-small"
base_url = ""  # e.g., "http://ollama:11434" for Docker

[llm]
provider = "openai"
model = "gpt-4o-mini"
base_url = ""  # e.g., "http://ollama:11434" for Docker
```

---

## Commit Messages

Use conventional commits format: `<type>(<scope>): <subject>`

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Scopes:** `adapters`, `config`, `pipelines`, `docker`, `docs`

**Examples:**
```
feat(adapters): add HuggingFace embedding provider
fix(llm): set stream=False in Ollama API requests
docs(readme): update Docker commands
refactor(config): simplify path resolution
test(pipelines): add unit tests for ingestion
```

**Rules:**
- Subject line: max 72 chars, lowercase, no period
- Use imperative mood ("add" not "added")
- Body: explain what and why (not how)
- Reference issues: `Closes #123`
