# Module 3: APIs and Runtime

## Layer 1: API (`app/`)

The API layer is the **top of the stack** — the HTTP boundary that the frontend, curl, and any other client talks to.
It consists of **two independent FastAPI applications** that share no runtime state.

| | Retrieval API | Ingestion API |
|---|---|---|
| **File** | `app/retrieval/main.py` | `app/ingestion/main.py` |
| **Port** | `:8000` | `:8001` |
| **Job** | Answer queries, manage models | Upload PDFs, trigger indexing |
| **Reads from** | Milvus (via pipeline) | Filesystem (`data/pdfs/`) |
| **Writes to** | State dir (eval jobs) | Milvus (via pipeline), filesystem |

They can be deployed, scaled, and restarted independently. The only coupling is that ingestion writes to Milvus and retrieval reads from it.

### Retrieval API Routes (`:8000`)

| Route | Purpose |
|---|---|
| `POST /query` | The main endpoint. Accepts a question, returns an LLM answer + source chunks. |
| `GET /health` | Returns runtime state (`healthy`, `degraded`, etc.), whether the pipeline is loaded, and whether documents exist. |
| `GET /config` | Reads the currently active embedding + LLM config from the runtime. |
| `POST /config` | Hot-swaps the active embedding or LLM. Delegates to `runtime.reload()`. |
| `GET /providers` | Live-scans all configured providers (Ollama, OpenAI, NIM) and returns available models for embedders, LLMs, and vision. |
| `GET /evaluate/{job_id}` | Polls async evaluation results (RAGAS metrics) for a previous query. |
| `GET /metrics` | Prometheus metrics endpoint. |

> **Note:** `POST /config` handles the state machine: HEALTHY → RELOADING → HEALTHY or DEGRADED.

**Two Query Modes:**
1. **Default mode:** Uses the pre-warmed pipeline. Fast path for normal usage.
2. **Stateless mode:** Builds a one-off pipeline if `llm` or `embedding` is passed in the request body.
   Used for A/B testing without restarting the service.

### Ingestion API Routes (`:8001`)

| Route | Purpose |
|---|---|
| `POST /upload` | Accepts PDF files via multipart form. Returns 409 if ingestion is already running. |
| `GET /status` | Reads ingestion progress from the atomic status file in `state/`. |
| `GET /health` | Simple liveness check. |
| `GET /files` | Lists uploaded PDFs in `data/pdfs/`. |
| `DELETE /documents/{filename}` | Deletes a PDF from disk **and** removes its vectors from Milvus. |
| `POST /config` | Swaps the default embedding for future ingestion jobs. |
| `POST /reindex` | Triggers a full re-index of all previously uploaded documents. |
| `POST /ingest/target` | Triggers a targeted ingest for a specific extraction/embedding permutation. |
| `POST /status/index` | Checks whether a given collection (by embedding + vision model) has documents in Milvus. |
| `GET /metrics` | Prometheus metrics endpoint. |

> **Note — `POST /upload`:** Validates filenames (path traversal prevention), checks file sizes, deduplicates, saves to `data/pdfs/`, then kicks off ingestion as a background task.

### Cross-Cutting Concerns (Middleware & Auth)

Both apps share the same middleware stack, wired up in order:

1. **CORS** — Origins come from `config.toml` via `get_frontend_origins()`, so the frontend on `:3000` can call the APIs.
2. **Structured Logging** (`src/structured_logging.py`) — Adds request IDs, timing, and structured context to every log line.
3. **Metrics** (`src/metrics.py`) — Collects Prometheus-style request metrics; exposed at `GET /metrics`.
4. **API Key Auth** (`src/auth.py`) — A `Depends(verify_api_key)` dependency that checks the
   `RecRAG-API-Key` header on protected routes. Health and metrics endpoints are excluded.

### Request/Response Models (`src/models/api.py`)

All Pydantic v2 models live in a **shared** `src/models/api.py` file. This is the contract between backend and
frontend — if you change a model shape here, `frontend/src/lib/api/` must stay in sync. Key models:

- **`QueryRequest`** / **`QueryResponse`** — The query contract. Response includes `response` (LLM text),
  `context` (list of `ContextItem` with text, source, distance), and `eval_job_id`.
- **`UploadResponse`** / **`StatusResponse`** — Ingestion contract.
- **`ConfigUpdateRequest`** / **`SetConfigResponse`** — Config hot-swap contract.
- **`ProvidersResponse`** — Nested dict of `ProviderInfo` (available, models, reason) per provider per category (embedders, llms, vision).
- **`ExtractionMode`** — StrEnum: `text_only` or `vision_assisted`.

### Key Design Patterns

1. **Stateless routes, stateful runtime** — Routes themselves hold no state. The `RetrievalRuntime` and
   `IngestionRuntime` are injected via FastAPI's `Depends()` and live on `app.state`. This makes routes
   easy to test and reason about.
2. **Background tasks for long work** — Both `/query` (evaluation) and `/upload` (ingestion) use
   `BackgroundTasks` so the HTTP response returns immediately while expensive work runs in the background.
3. **Validation at the boundary** — Input validation (`validate_query`, `validate_filename`,
   `validate_file_size`) happens in the route handler before anything reaches the pipeline. Path traversal
   attacks on filenames are caught here.
4. **Error mapping** — Raw exceptions from pipelines and adapters are caught in the route and mapped to
   appropriate HTTP status codes (400 for bad input, 409 for conflict, 500 for unexpected errors,
   503 when the runtime isn't ready).
5. **Dual query modes** — The `/query` route supports both a fast path (use the pre-warmed pipeline)
   and a stateless path (build a one-off pipeline with different model overrides), enabling A/B testing
   without restarting the service.

## Layer 2: Runtime (`src/runtime/`)

The API layer is stateless. The **Runtime** owns the pipeline instances — warming them up, keeping them alive,
handling hot-reloads, and shutting them down.

### Lifecycle State Machine

```text
STARTING ──warm()──▶ HEALTHY ──reload()──▶ RELOADING ──▶ HEALTHY
    │                    │                      │
    └── failure ────────▶ DEGRADED ◀── failure ─┘
```

The runtime acts as an async context manager (`RetrievalRuntime.acquire()`) to handle concurrent reads
while allowing safe, lock-managed pipeline reloads.

Move on to [Module 4: Pipelines, Adapters, and Stores](04_pipelines_and_adapters.md).
