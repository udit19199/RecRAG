# AGENTS.md - Development Guidelines for RecRAG

Guidelines for agentic coding agents working in this repository.

## Project Overview

RecRAG is a Retrieval-Augmented Generation pipeline with separate ingestion and retrieval containers.

**Tech Stack:** Python 3.12+, llama-index, FAISS, Streamlit, uv, Docker, OpenAI/Ollama/NVIDIA NIM

---

## Build / Lint / Test Commands

```bash
# Installation
uv sync

# Testing
uv run pytest                                     # Run all tests
uv run pytest tests/test_ingest.py                # Run single file
uv run pytest tests/test_ingest.py::test_name     # Run single test
uv run pytest -k "pattern"                        # Run tests matching pattern

# Linting & Formatting
uv run ruff check .                               # Lint
uv run ruff format .                              # Format
uv run ruff check --fix .                         # Auto-fix

# Type Checking
uv run mypy backend/

# Local Development
uv run python backend/watch.py                    # File watcher
uv run streamlit run backend/app.py               # Streamlit UI
uv run python backend/ingest.py --force           # One-time ingestion

# Docker
docker-compose build                               # Build images
docker-compose up -d                               # Start all services
docker-compose down                                # Stop containers
```

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

### Adapter Pattern (`backend/src/adapters/`)
- `BaseEmbedder` and `BaseLLM` are abstract base classes
- Factory functions in `__init__.py`: `create_embedder()`, `create_llm()`

### Configuration (`backend/src/config.py`)
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
├── backend/
│   ├── src/
│   │   ├── config.py         # Config loading & path resolution
│   │   ├── core.py           # Document processing & vector store
│   │   ├── pipelines.py      # Ingestion & retrieval pipelines
│   │   └── adapters/         # LLM & embedding providers
│   ├── app.py                # Streamlit UI
│   ├── ingest.py             # CLI tool
│   └── watch.py              # File watcher daemon
├── Dockerfile                # Docker image definition
├── docker-compose.yml        # Multi-container orchestration
├── config.toml               # Application config
├── .env                      # Environment variables (not in git)
├── data/pdfs/                # PDF uploads (not in git)
└── storage/                  # FAISS index (not in git)
```

---

## Configuration

### File Organization
- **`config.toml`**: All configuration settings (provider, model, URLs, chunk sizes, etc.)
- **`.env`**: API keys only (sensitive information)

### Environment Variables (`.env`)
Only sensitive values:
- `OPENAI_API_KEY`: Required for OpenAI provider
- `NVIDIA_API_KEY`: Required for NVIDIA NIM provider (starts with "nvapi-")

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
