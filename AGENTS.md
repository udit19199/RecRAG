# AGENTS.md - RecRAG Operating Guide

Use this file as the primary guide for agentic coding work in this repository.

## Product Vision

RecRAG is a **multi-model RAG pipeline recommendation system**. It helps users
find the right RAG pipeline for their use case. It does **not** deploy that pipeline
into RecRAG — the deliverable is a **JSON blueprint** (architecture, models, params,
cost, benchmark evidence) the user implements elsewhere.

### Ideal User Intake Flow

Collect these inputs before recommending:

1. **Use case** — What will the RAG system be used for?
2. **Audience** — Who will use it (role, expertise, volume)?
3. **Document modality** — Type of content queried: text-heavy PDFs, scanned docs,
   image/chart-heavy, video, mixed (not user interaction mode).
4. **Budget** — Monthly spend ceiling (tokens, cloud, storage).
5. **Other constraints** — Citations, latency, compliance, languages, data sensitivity.

Example:

> A law firm wants employees to query procedures and compliance. Budget: $1,500/month.
> Citations mandatory. Junior staff are primary users. Text-heavy policy PDFs.

### How Recommendation Works

1. **Intake** (wizard or chat) → `Requirements`
2. **Shortlist** 3–5 `PipelineCandidate`s (min 3, max 5 during research)
3. **Index** each candidate on the user's corpus; **benchmark** with the same suite
4. **Deliver** one recommendation — user does not pick among options
5. **Export** JSON blueprint only after benchmark (fast path without corpus is UI-only)

Ingestion variance is **use-case-relative**: text-heavy + citations may have few ingest
options (optimize retrieval); image-heavy or graph cases need more ingest choices.

### Design documentation (read before implementing)

| Document | Purpose |
|----------|---------|
| [docs/recommendation/DESIGN_DECISIONS.md](docs/recommendation/DESIGN_DECISIONS.md) | **Resolved** decisions (32+ entries) — authoritative |
| [docs/recommendation/OPEN_QUESTIONS.md](docs/recommendation/OPEN_QUESTIONS.md) | **Unresolved** TBD items — check before guessing |
| `.cursor/plans/rag_recommendation_system_3de486ca.plan.md` | Implementation phases and file map |

**When DESIGN_DECISIONS and this file conflict, DESIGN_DECISIONS wins.** Update
DESIGN_DECISIONS when closing an item from OPEN_QUESTIONS.

### Documentation requirement

**Every meaningful code change must document why it was done.** This is mandatory for
orchestration, benchmark, intake, and schema work.

- **Closing a TBD:** add entry to DESIGN_DECISIONS; remove from OPEN_QUESTIONS
- **Non-obvious implementation choice:** brief comment in code or module docstring
  referencing the decision ID (e.g. `D22 — user-driven collection retention`)
- **Commits / PRs:** state the *why*, not only the *what*
- **Research findings:** `docs/research/findings/` during ideation (archive before prod)

Do not merge orchestration behavior changes without updating docs when the change
reflects a new or clarified product decision.

### Architecture summary

| Component | Location | Notes |
|-----------|----------|-------|
| Orchestrator API | `app/orchestrator/` `:8002` | Intake, runs, benchmark, export, history |
| Orchestration engine | `src/orchestration/` | constraints, resolver, customizer, cost |
| Benchmark | `src/benchmark/` | suite, runner, templates (later) |
| Ingestion API | `app/ingestion/` `:8001` | Unchanged; called by Orchestrator |
| Retrieval API | `app/retrieval/` `:8000` | Unchanged; called by Orchestrator |
| History DB | Postgres + Alembic | `DATABASE_URL` in `.env` |
| Auth | Clerk B2B | Organizations = `workspace_id`; lazy sync, no webhooks v1 |
| Active `RagArchitecture` | `naive`, `citation`, `multimodal` | `graph`, `agentic` commented until built or removed |
| Pricing | `data/model_pricing.toml` | Joined with live `/providers`; flagship = most expensive |

### What Exists Today vs Target

| Area | Today | Target |
|------|-------|--------|
| Ingestion / retrieval | Working RAG, Milvus, multi-provider | Same; used internally for benchmark |
| User intake | Generic onboarding shell | Dual intake → `Requirements` |
| Recommendation | Not implemented | Orchestrator shortlist → benchmark → JSON export |
| History | Not implemented | Postgres per `workspace_id`; lookup past runs |
| Deploy winner to RecRAG | N/A | **Out of scope** — blueprint export only |
| Evaluation | `src/evaluation/ragas_eval.py` | Constraint-weighted benchmark scoring |

Research-only behaviors (experiment JSON, stretch backfill, markdown findings) are
temporary — archive before prod.

## Repository Snapshot
- Backend: Python 3.12, FastAPI, uv, pytest, Ruff, mypy, Milvus, Postgres (planned).
- Frontend: Next.js 16, React 19, TypeScript, Tailwind, Biome, Clerk B2B (planned).
- Backend code: `src/`, `app/`, `jobs/`.
- Frontend code: `frontend/`.
- Tests: `tests/`.

## Working Principles
- Make small, focused changes.
- Preserve existing architecture unless the task clearly needs refactoring.
- Keep backend, frontend, and config changes aligned.
- Orchestration logic lives in `src/orchestration/`, not in UI conditionals.
- Orchestrator API is a **separate service** (`:8002`) — do not add orchestration routes to retrieval.
- Treat pipeline configs as data (`PipelineSpec`), not ad-hoc `config.toml` edits per user.
- Read DESIGN_DECISIONS before implementing; document why on every meaningful change.
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

Orchestrator (`:8002`) — planned; see OPEN_QUESTIONS INF-1.

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
- `docs/recommendation/DESIGN_DECISIONS.md` — resolved recommendation system decisions
- `docs/recommendation/OPEN_QUESTIONS.md` — unresolved TBD items
- `k8s/`: Kubernetes manifests (Kustomize base + overlays for dev/prod)
- `docs/KUBERNETES_DEPLOYMENT.md`: K8s deployment guide
- `PRODUCT.md`, `DESIGN.md`: product voice and visual system (not pipeline logic)
- `app/ingestion/main.py`: upload, status, config, reindex, and delete routes
- `app/retrieval/main.py`: query, config, provider, and health routes
- `src/runtime/`: pipeline lifecycle and warmup logic
- `src/pipelines/`: ingestion and retrieval pipelines
- `src/adapters/`: LLM, embedding, and vision adapters
- `src/evaluation/`: faithfulness, relevancy, context precision/recall evaluators
- `src/providers.py`: provider model catalogs
- `src/loaders.py`, `src/splitters.py`, `src/stores.py`: document → chunk → vector path
- `src/models/`: shared API and domain models
- `src/auth.py`: API key auth for ingestion/retrieval (Orchestrator uses Clerk)
- `state/`: runtime status and evaluation job JSON files
- `frontend/app/(main)/generate/`: recommendation generator form / intake wizard
- `frontend/src/lib/api/`: API client and typed request/response helpers
- `tests/`: pytest suite
- `scripts/deploy.sh`: single-command deploy script

### Planned / Not Yet Present
- `app/orchestrator/` — Orchestrator API `:8002`
- `src/orchestration/` — engine (constraints, resolver, customizer, cost)
- `src/benchmark/` — suite, runner
- `data/model_pricing.toml`, `data/recommendation_rules.toml`
- `alembic/` — Postgres migrations
- `frontend/src/features/intake/`, `frontend/src/features/recommendation/`

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
- Recommendation domain: `Requirements`, `PipelineSpec`, `RagArchitecture` (see DESIGN_DECISIONS glossary)

### Error Handling
- Raise specific exceptions where practical.
- In API routes, raise `HTTPException` with actionable messages.
- Avoid bare `except:`.
- Do not leak secrets in errors or logs.
- Preserve async/background behavior unless the task explicitly changes it.

### Docs and Comments
- Public functions and classes should have concise docstrings.
- Describe behavior and constraints, not line-by-line implementation.
- Reference DESIGN_DECISIONS IDs when implementing non-obvious orchestration behavior.

## Backend Patterns
- Keep route schemas near the route code.
- Reuse shared config and pipeline helpers.
- If API contracts change, update backend models and `frontend/src/lib/api/` together.
- Use background tasks for long-running ingestion and benchmark work.
- Do not block request handlers with expensive pipeline work when existing code uses background execution.
- Use `anyio.open_file` for async-safe file I/O in API routes.
- Use `asyncio.to_thread` for blocking filesystem operations.
- Orchestrator calls ingestion/retrieval with `REC_RAG_API_KEY`; user auth via Clerk JWT on Orchestrator only (v1).

## Frontend Patterns
- Follow the App Router structure in `frontend/app/`.
- Use typed props and typed API responses.
- Keep shared UI primitives in `frontend/src/components/ui/`.
- Keep API calls centralized in `frontend/src/lib/api/`.
- Use function components and hooks.
- Preserve established UI patterns unless a task explicitly calls for redesign.
- Keep `pnpm build` green after meaningful frontend or API client changes.
- Clerk B2B: pass session token to Orchestrator; `org_id` scopes runs.

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
- Include **why** in commit body for orchestration/recommendation changes.

## Agent Checklist
Before finishing:
- imports are grouped correctly
- types are updated where contracts changed
- backend and frontend API shapes still match
- DESIGN_DECISIONS / OPEN_QUESTIONS updated if a decision was made or closed
- meaningful changes document **why** (comment, commit, or decision entry)
- the smallest relevant lint/test/build commands have run
- local-only files were not staged by accident

Keep this file practical and repository-specific when updating it.
