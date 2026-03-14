# AGENTS.md - Guidance for Coding Agents

Default operating guide for agentic coding tools working in `RecRAG`.

## Project Summary

RecRAG is a Retrieval-Augmented Generation app with:
- retrieval FastAPI routes in `api/retrieval/`
- ingestion FastAPI routes in `api/ingestion/`
- shared Python packages in `src/`
- CLI entrypoints in `cli/`
- a Next.js frontend in `frontend/`
- tests in `tests/`

Backend stack: Python 3.12, `uv`, FastAPI, pytest, Ruff, mypy, Milvus.
Frontend stack: Next.js, React 19, TypeScript, Tailwind, Biome.

## Repo-Specific Rule Files

Checked locations:
- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

If any are added later, treat them as higher-priority instructions and merge them with this file.

## Working Norms

- Prefer small, focused changes.
- Preserve architecture unless the task requires refactoring.
- Keep backend and frontend API contracts aligned.
- Do not commit local-only tooling files unless requested.
- Avoid destructive git commands.
- Keep secrets in `.env`, not in tracked config.

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
- Do not move `PYTHONPATH` into `.env`.
- Keep API keys in `.env` only.

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

Important details:
- `make lint` covers backend Ruff only.
- Frontend lint/format use Biome from `frontend/package.json`.
- After frontend changes, run at least `cd frontend && pnpm lint`.
- Prefer `cd frontend && pnpm build` after meaningful UI or API-client changes.

## Test Commands

Primary entrypoint:

```bash
make test
uv run pytest
```

Run a single file:

```bash
uv run pytest tests/test_ingest.py
uv run pytest tests/test_evaluation/test_ragas_eval.py
```

Run a single test:

```bash
uv run pytest tests/test_ingest.py::test_name
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

Type checking note:
- `mypy` is the standard command here.
- `pyright` is configured in `pyproject.toml`, but not wired into `make`.

## Key Paths

- `api/ingestion/main.py`: ingestion API routes and reindex/status endpoints
- `api/retrieval/main.py`: retrieval API, provider/config routes, eval job polling
- `src/adapters/`: LLM and embedding providers
- `src/pipelines/`: ingestion and retrieval pipelines
- `src/evaluation/`: RAGAS helpers
- `src/stores/`: vector store integration
- `frontend/src/lib/api.ts`: frontend API client
- `frontend/src/components/ui/`: shared UI primitives
- `tests/`: pytest suite

## Python Style Guidelines

### Imports

Use three groups with blank lines between them:
1. standard library
2. third-party packages
3. local application imports

Prefer explicit imports over wildcard imports.

### Formatting

- Follow Ruff formatting.
- Keep functions and route handlers easy to scan.
- Prefer multiline calls when argument lists get dense.
- Add comments only when intent is not obvious.

### Types

- Add type hints for public functions, methods, and important helpers.
- Prefer built-in generics like `list[str]` and `dict[str, Any]`.
- Prefer `X | None` over `Optional[X]`.
- Keep FastAPI request/response models explicitly typed.
- Preserve Pydantic model structure unless the API contract changes.

### Naming

- functions and variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`
- private helpers and attributes: leading underscore

### Paths And Config

- Prefer `pathlib.Path` over raw strings for filesystem work.
- Use helpers from `src/config.py` for config loading and path resolution.
- Keep non-sensitive defaults in `config.toml`.
- Keep secrets and credentials in `.env` only.

### Error Handling

- Use specific exception types where practical.
- In API routes, raise `HTTPException` with actionable messages.
- Avoid bare `except:` blocks.
- Only suppress exceptions intentionally and document why.
- Do not leak secrets in error messages.

### Docstrings

- Public functions and classes should have concise docstrings.
- Explain behavior and constraints, not line-by-line implementation.

## Backend Patterns

- Keep route schemas near route code using Pydantic models.
- Reuse shared pipeline/config helpers instead of duplicating setup logic.
- If changing route behavior, update backend models and `frontend/src/lib/api.ts` together.
- Background evaluation uses async job polling; preserve that contract carefully.
- CORS is permissive for development; do not change it unless the task is about deployment or security.

## Frontend Style Guidelines

- Follow the App Router structure in `frontend/app/`.
- Use typed props and typed API responses.
- Keep shared primitives in `frontend/src/components/ui/`.
- Prefer existing utilities like `cn` instead of duplicating class-merging logic.
- Keep API calls centralized in `frontend/src/lib/api.ts`.
- Use function components and hooks.

Frontend tooling details:
- Biome handles lint and format.
- Tailwind utility classes are standard in this codebase.
- Preserve established UI patterns unless a task explicitly calls for redesign.

## Testing Expectations

- Add or update tests when changing shared pipeline, adapter, or evaluation behavior.
- For API contract changes, validate both backend behavior and frontend types.
- Start with the smallest relevant test run, then broaden as needed.
- If you cannot run full verification, say what was run and what remains unverified.

## Commit Guidance

- Use conventional commits.
- Common types here: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Keep subjects lowercase, imperative, and under 72 characters.
- Split commits by intent when UI, API, and config changes are logically separate.

Examples: `feat(frontend): add app shell and comparison workspace`, `refactor(retrieval): make query evaluation always asynchronous`, `chore(config): update local model and ui defaults`

## Agent Checklist

Before finishing:
- imports are grouped correctly
- types are updated where contracts changed
- backend and frontend API shapes still match
- the smallest relevant lint/test/build commands have run
- local-only files were not staged by accident

Keep this file practical and repository-specific when updating it.
