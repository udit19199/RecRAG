# AGENTS.md - Development Guidelines for RecRAG

Guidelines for agentic coding agents working in this repository.

## Project Overview

RecRAG is a Retrieval-Augmented Generation pipeline with separate ingestion and retrieval containers.

**Tech Stack:** Python 3.12+, llama-index, FAISS, Streamlit, uv, Docker, OpenAI/Ollama

**Architecture:**
```
Ollama Container → ollama-init (pulls models) → ingestion + retrieval containers
                                                    ↓
                                            Shared volumes (data/, storage/)
```

---

## Build / Lint / Test Commands

### Installation
```bash
uv sync                          # Install dependencies
```

### Docker
```bash
docker-compose -f docker/docker-compose.yml build          # Build images
docker-compose -f docker/docker-compose.yml up -d          # Start all services
docker-compose -f docker/docker-compose.yml logs -f ingestion  # View logs
docker-compose -f docker/docker-compose.yml down           # Stop containers
```

### Local Development
```bash
uv run python backend/watch.py                    # File watcher
uv run streamlit run backend/app.py               # Streamlit UI
uv run python backend/ingest.py --force           # One-time ingestion
```

### Testing
```bash
uv run pytest                                     # Run all tests
uv run pytest tests/test_ingest.py                # Run single file
uv run pytest tests/test_ingest.py::test_name     # Run single test
uv run pytest -k "pattern"                        # Run tests matching pattern
```

### Linting & Formatting
```bash
uv run ruff check .                               # Lint
uv run ruff format .                              # Format
uv run ruff check --fix .                         # Auto-fix
```

### Type Checking
```bash
uv run mypy backend/
```

---

## Code Style Guidelines

### Import Order (separate groups with blank line)
1. Standard library  2. Third-party  3. Local application

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
- Write docstrings for all public functions and classes

```python
def embed(self, text: str) -> list[float]:
    """Generate embedding for a single text.
    
    Args:
        text: The text to embed.
    
    Returns:
        List of floats representing the embedding.
    """
```

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
- **Always set `"stream": False`** in request payloads to get single JSON response
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
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── config.toml               # Application config
├── .env                      # Environment variables (not in git)
├── data/pdfs/                # PDF uploads (not in git)
└── storage/                  # FAISS index (not in git)
```

---

## Environment Variables

Required in `.env`:
- `EMBEDDING_PROVIDER` / `LLM_PROVIDER`: "openai" or "ollama"
- `EMBEDDING_MODEL` / `LLM_MODEL`: Model name
- `EMBEDDING_BASE_URL` / `LLM_BASE_URL`: For Ollama, use `http://ollama:11434`
- `OPENAI_API_KEY`: Required if using OpenAI

---

## Commit Messages

Use conventional commits format:

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style (formatting, whitespace) |
| `refactor` | Code refactoring |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `perf` | Performance improvements |
| `ci` | CI/CD changes |

### Scopes
Use module or component name: `adapters`, `config`, `pipelines`, `docker`, `docs`

### Examples
```
feat(adapters): add HuggingFace embedding provider

fix(llm): set stream=False in Ollama API requests

docs(readme): update Docker commands for new architecture

refactor(config): simplify path resolution logic

test(pipelines): add unit tests for ingestion pipeline
```

### Rules
- Subject line: max 72 characters, lowercase, no period at end
- Use imperative mood ("add" not "added" or "adds")
- Body: explain what and why (not how)
- Reference issues in footer: `Closes #123`

---

## Common Issues

| Issue | Solution |
|-------|----------|
| Memory error with LLM | Increase Docker memory or use smaller model |
| 404 on `/api/generate` | Check model name matches exactly (e.g., `granite3.1-moe:1b`) |
| JSON parsing error | Ensure `"stream": False` in Ollama requests |
| Env vars not substituted | Ensure `.env` exists and containers restarted |
