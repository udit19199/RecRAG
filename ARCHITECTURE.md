# RecRAG Architecture Guide

RecRAG is a multi-service RAG application with a recommendation orchestrator on top.

## Services

| Service | Port | Location | Responsibility |
|---------|------|----------|----------------|
| Retrieval API | `8000` | `app/retrieval/` | Query, config, providers, evaluation jobs |
| Ingestion API | `8001` | `app/ingestion/` | Upload, status, reindex, delete |
| Orchestrator API | `8002` | `app/orchestrator/` | Intake, recommendation runs, export |
| Frontend | `3000` | `frontend/` | Chat, compare, generate, history |

## Backend layers

| Layer | Location | Notes |
|-------|----------|-------|
| API routes | `app/` | FastAPI handlers and request/response models |
| Orchestration | `src/orchestration/` | Requirements, run lifecycle, benchmark orchestration |
| Pipelines | `src/pipelines/` | Ingestion and retrieval flows |
| Adapters | `src/adapters/` | LLM, embedding, and vision providers |
| Storage | `src/stores.py`, `config.toml` | Milvus Lite by default; Docker Milvus optional |

## Common tasks

| "I want to..." | Look here |
|----------------|-----------|
| Add a new LLM provider | `src/adapters/` + register in `__init__.py` |
| Change how chunks are created | `src/splitters.py` |
| Change the RAG prompt template | `config.toml` → `[retrieval] context_template` |
| Add a retrieval or ingestion route | `app/retrieval/main.py` or `app/ingestion/main.py` |
| Change recommendation run behavior | `src/orchestration/run_service.py` |
| Change Milvus connection settings | `config.toml` → `[storage]` |

## Local setup

```bash
make setup
make dev
```

For Docker-based deployment:

```bash
cp .env.example .env
make docker-up
```

See `README.md` for the full command list and `AGENTS.md` for contributor conventions.
