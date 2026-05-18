# AGENTS.md - RecRAG Operating Guide

Use this file as the primary guide for agentic coding work in this repository.

## Repository Snapshot
- Backend: Python 3.12, FastAPI, uv, pytest, Ruff, mypy, Milvus.
- Frontend: Next.js 16, React 19, TypeScript, Tailwind, Biome.
- Backend code: `src/`, `app/`, `jobs/`.
- Frontend code: `frontend/`.
- Tests: `tests/`.

## Working Principles
- Make small, focused changes.
- Preserve existing architecture unless the task clearly needs refactoring.
- Keep backend, frontend, and config changes aligned.
- Do not overwrite unrelated user changes.
- Avoid destructive git commands.
- Keep secrets in `.env`, not in tracked files.
- Do not move API keys into `config.toml`.
- Prefer the smallest correct change over adding new abstractions.
- API keys are popped from kwargs before passing to base adapters (implemented lazily in NIM, eagerly in OpenAI/Ollama).
- Retries use `urllib3.util.Retry` with idempotent-method-only.
- Adapter timeouts come from `config.toml`, not hardcoded.
- Token-counting fallback is configurable via `retrieval.tokenizer_fallback`.

## Setup and Common Commands
Run from the repository root unless noted otherwise.

### Full Setup
- `make install`
- `make setup`
- `make dev`

### App Processes
- `make retrieval` — Start retrieval API on `:8000`
- `make ingestion` — Start ingestion API on `:8001`
- `make frontend` — Start frontend on `:3000`
- `make ingest` — Run `jobs/ingest.py --force`
- `make evaluate` — Run `jobs/evaluate.py`

### Infra
- `make infra-up` — Start local Milvus stack
- `make infra-down` — Stop local Milvus stack
- `make infra-logs` — Show Milvus logs
- `make docker-up` — Start full stack in Docker
- `make docker-down` — Stop full stack

### Deploy
- `make deploy` — Build, import into k3s, and deploy
- `make deploy-dev` — Deploy with dev overlay (lightweight)

Notes:
- `make dev` starts retrieval on `:8000`, ingestion on `:8001`, and frontend on `:3000`.
- `make test` runs backend tests only.
- The deploy script at `scripts/deploy.sh` uses `--from-env-file` for secrets, validates env without sourcing, and runs smoke tests after deploy.

## Build, Lint, and Format
### Backend
- `make lint` / `make lint-backend`
- `make format` / `make format-backend`
- `make typecheck`
- `make check` — lint, typecheck, test
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src/`

### Frontend
- `cd frontend && pnpm lint`
- `cd frontend && pnpm format`
- `cd frontend && pnpm build`

Notes:
- Frontend lint and format use Biome.
- Run `pnpm build` after meaningful UI or API client changes.

## Test Commands
### Primary
- `make test`
- `uv run pytest`

### Single File
- `uv run pytest tests/test_status.py`
- `uv run pytest tests/test_adapters/test_llm.py`

### Single Test
- `uv run pytest tests/test_status.py::test_write_status_is_atomic`

### Targeted Runs
- `uv run pytest -k "pattern"`
- `uv run pytest -x`
- `uv run pytest -q`

Notes:
- Use the smallest relevant test first.
- There is no frontend test script; use lint/build for verification.
- If a command fails, fix the smallest surface area possible before broadening the run.

## Key Paths
- `k8s/`: Kubernetes manifests (Kustomize base + overlays for dev/prod)
- `docs/KUBERNETES_DEPLOYMENT.md`: K8s deployment guide
- `app/ingestion/main.py`: upload, status, config, reindex, and delete routes.
- `app/retrieval/main.py`: query, config, provider, and health routes.
- `src/runtime/`: pipeline lifecycle and warmup logic.
- `src/pipelines/`: ingestion and retrieval pipelines.
- `src/adapters/`: LLM, embedding, and vision adapters.
- `src/loaders.py`: document loaders.
- `src/splitters.py`: text chunking.
- `src/stores.py`: vector store integration.
- `src/utils/status.py`: ingestion status persistence.
- `src/config.py`: config loading and path helpers.
- `state/`: runtime status and evaluation job JSON files.
- `frontend/app/`: App Router pages.
- `frontend/src/lib/api/`: API client and typed request/response helpers.
- `tests/`: pytest suite.
- `scripts/deploy.sh`: single-command deploy script.

## Python Style Guidelines
### Imports
- Use three groups with blank lines: standard library, third-party, local.
- Prefer explicit imports.
- Avoid wildcard imports.
- Keep import ordering stable and Ruff-friendly.

### Formatting
- Follow Ruff formatting.
- Keep route handlers and helpers short and readable.
- Use multiline calls when argument lists get dense.
- Add comments only when intent is not obvious.
- Prefer `pathlib.Path` for filesystem work.

### Types
- Add type hints for public functions, methods, and important helpers.
- Prefer built-in generics like `list[str]` and `dict[str, Any]`.
- Prefer `X | None` over `Optional[X]` when practical.
- Keep FastAPI request and response models explicit.
- Use `Any` sparingly and locally.

### Naming
- functions and variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`
- private helpers and attributes: leading underscore

### Error Handling
- Raise specific exceptions where practical.
- In API routes, raise `HTTPException` with actionable messages.
- Avoid bare `except:`.
- Do not leak secrets in errors or logs.
- Preserve async/background behavior unless the task explicitly changes it.

### Docs and Comments
- Public functions and classes should have concise docstrings.
- Describe behavior and constraints, not line-by-line implementation.
- Keep comments brief and rare.

## Backend Patterns
- Keep route schemas near the route code.
- Reuse shared config and pipeline helpers.
- If API contracts change, update backend models and `frontend/src/lib/api.ts` together.
- Use background tasks for long-running ingestion work.
- Do not block request handlers with expensive pipeline work when existing code uses background execution.
- Use `anyio.open_file` for async-safe file I/O in API routes.
- Use `asyncio.to_thread` for blocking filesystem operations.

## Frontend Patterns
- Follow the App Router structure in `frontend/app/`.
- Use typed props and typed API responses.
- Keep shared UI primitives in `frontend/src/components/ui/`.
- Keep API calls centralized in `frontend/src/lib/api/`.
- Use function components and hooks.
- Preserve established UI patterns unless a task explicitly calls for redesign.
- Keep `pnpm build` green after meaningful frontend or API client changes.

## Testing Expectations
- Add or update tests when changing shared pipeline, adapter, or status behavior.
- For API contract changes, validate both backend behavior and frontend types.
- Start with the smallest relevant test run, then broaden as needed.
- If full verification is not possible, say what ran and what remains unverified.

## Commit Guidance
- Use conventional commits.
- Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
- Keep subjects lowercase, imperative, and under 72 characters.
- Example: `fix(status): write ingestion state atomically`

## Agent Checklist
Before finishing:
- imports are grouped correctly
- types are updated where contracts changed
- backend and frontend API shapes still match
- the smallest relevant lint/test/build commands have run
- local-only files were not staged by accident

Keep this file practical and repository-specific when updating it.
