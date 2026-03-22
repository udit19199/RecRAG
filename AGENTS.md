# AGENTS.md - RecRAG Operating Guide

Use this file as the primary guide for agentic coding work in this repository.

## Repository Snapshot

- Backend: Python 3.12, FastAPI, uv, pytest, Ruff, mypy, Milvus.
- Frontend: Next.js, React 19, TypeScript, Tailwind, Biome.
- Main backend code: `src/`, `api/`, `cli/`.
- Frontend code: `frontend/`.
- Tests: `tests/`.

## Repo-Specific Rule Files

- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

If any appear later, treat them as higher-priority instructions and merge them with this guide.

## Working Principles

- Make small, focused changes.
- Preserve existing architecture unless the task clearly needs refactoring.
- Keep backend, frontend, and config changes aligned.
- Do not overwrite unrelated user changes.
- Avoid destructive git commands.
- Keep secrets in `.env`, not in tracked files.

## Setup And Dev Commands

Run from the repository root unless noted otherwise.

```bash
make install
make dev
make retrieval
make ingestion
make frontend
make ingest
```

Notes:
- `make dev` starts retrieval on `:8000`, ingestion on `:8001`, frontend on `:3000`.
- The Makefile uses `PYTHONPATH=src uv run --env-file .env ...`.
- Do not move API keys into `config.toml`.

Manual equivalents:

```bash
PYTHONPATH=src uv run --env-file .env uvicorn api.retrieval.main:app --port 8000 --reload
PYTHONPATH=src uv run --env-file .env uvicorn api.ingestion.main:app --port 8001 --reload
cd frontend && pnpm dev
```

## Build, Lint, And Format

Backend:

```bash
make lint
make format
make typecheck
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

Frontend:

```bash
cd frontend && pnpm lint
cd frontend && pnpm format
cd frontend && pnpm build
```

Guidance:
- `make lint` covers backend Ruff only.
- Frontend lint/format use Biome.
- Run `pnpm build` after meaningful UI or API-client changes.

## Test Commands

Primary entrypoint:

```bash
make test
uv run pytest
```

Run one file:

```bash
uv run pytest tests/test_status.py
uv run pytest tests/test_evaluation/test_ragas_eval.py
```

Run one test:

```bash
uv run pytest tests/test_status.py::test_write_status_is_atomic
uv run pytest tests/test_evaluation/test_ragas_eval.py::test_name
```

Run by pattern:

```bash
uv run pytest -k "pattern"
```

Useful iteration flags:

```bash
uv run pytest -x
uv run pytest -q
```

## Key Paths

- `api/ingestion/main.py`: upload, status, reindex, config routes.
- `api/retrieval/main.py`: query, config, provider, health routes.
- `src/adapters/`: LLM and embedding providers.
- `src/pipelines/`: ingestion and retrieval pipelines.
- `src/utils/status.py`: ingestion status persistence.
- `src/config.py`: config loading and path helpers.
- `src/stores/`: vector store integration.
- `tests/`: pytest suite.

## Python Style Guidelines

### Imports

- Use three groups with blank lines: standard library, third-party, local.
- Prefer explicit imports.
- Avoid wildcard imports.

### Formatting

- Follow Ruff formatting.
- Keep route handlers and helpers readable and short.
- Use multiline calls when argument lists are dense.
- Add comments only when intent is not obvious.

### Types

- Add type hints for public functions, methods, and important helpers.
- Prefer built-in generics like `list[str]` and `dict[str, Any]`.
- Prefer `X | None` over `Optional[X]` when practical.
- Keep FastAPI request/response models explicitly typed.

### Naming

- functions and variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`
- private helpers and attributes: leading underscore

### Paths And Config

- Prefer `pathlib.Path` for filesystem work.
- Use helpers in `src/config.py` for config loading and path resolution.
- Keep non-sensitive defaults in `config.toml`.
- Keep secrets in `.env` only.

### Error Handling

- Raise specific exceptions where practical.
- In API routes, raise `HTTPException` with actionable messages.
- Avoid bare `except:` blocks.
- Do not leak secrets in errors or logs.

### Docstrings

- Public functions and classes should have concise docstrings.
- Describe behavior and constraints, not line-by-line implementation.

## Backend Patterns

- Keep route schemas near route code using Pydantic models.
- Reuse shared config and pipeline helpers.
- If changing a route contract, update backend models and `frontend/src/lib/api.ts` together.
- Preserve async/background task behavior unless the task explicitly changes it.

## Frontend Style Guidelines

- Follow the App Router structure in `frontend/app/`.
- Use typed props and typed API responses.
- Keep shared primitives in `frontend/src/components/ui/`.
- Keep API calls centralized in `frontend/src/lib/api.ts`.
- Use function components and hooks.

Frontend tooling:
- Biome handles lint and format.
- Tailwind utility classes are standard.
- Preserve established UI patterns unless a task explicitly calls for redesign.

## Testing Expectations

- Add or update tests when changing shared pipeline, adapter, or status behavior.
- For API contract changes, validate both backend behavior and frontend types.
- Start with the smallest relevant test run, then broaden as needed.
- If full verification is not possible, say what ran and what remains unverified.

## Commit Guidance

- Use conventional commits.
- Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Keep subjects lowercase, imperative, and under 72 characters.

Examples: `fix(status): write ingestion state atomically`, `docs(plan): remove completed roadmap items`

## Agent Checklist

Before finishing:
- imports are grouped correctly
- types are updated where contracts changed
- backend and frontend API shapes still match
- the smallest relevant lint/test/build commands have run
- local-only files were not staged by accident

Keep this file practical and repository-specific when updating it.
