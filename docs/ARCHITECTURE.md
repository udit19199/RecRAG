# RecRAG Architecture

A code-driven architecture document for the RecRAG RAG pipeline system.
Ongoing issues, technical debt, and future work are tracked in `docs/ISSUES.md`.

---

## 1. Overview

## What the System Does

RecRAG is a **RAG pipeline recommendation system**, which analyzes user context and constraints and **recommends suitable RAG pipeline configurations**.

The system focuses on **decision support**, not execution: it helps users choose defensible, industry-appropriate RAG pipelines that can later be deployed.

---

## Primary Use Cases

- **Pipeline Recommendation (Core Use Case)**  
  Recommend one or more RAG pipeline configurations based on:
  - Industry (Manufacturing / BFSI)
  - Expected document types (PDFs, manuals, policies)
  - Latency, cost, and compliance constraints  
  - Recommendations are explainable and do not require user data.
  - Provide rationale, trade-offs, and known limitations for each recommended pipeline.

- **Deployment Readiness Output**  
  Emit structured pipeline configurations (e.g., JSON/YAML) that can be consumed by downstream systems or teams to instantiate a running RAG pipeline.

---

## Non-Goals

- Running or hosting user RAG pipelines
- Document ingestion or question answering at recommendation time
- User-data-driven tuning or automated pipeline optimization
- Multi-tenant SaaS features (auth, billing, collaboration)
- End-user application UX

---

## Design Philosophy

- **Recommendation over execution**: advise, don’t run
- **Explainability over optimality**: defensible choices
- **Domain-aware defaults**: conservative pipelines for regulated industries
- **Zero user data dependency**: operates entirely on priors and constraints

---

## 2. Architecture

### Overall Architectural Style

**Microservices with shared storage.**

The system has evolved from a dual-container pattern to a microservices architecture with independent API services accessible via a Next.js frontend.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend (Port 3000)                     │
│  ┌─────────────────┐                       ┌─────────────────────────┐   │
│  │   / (Query)     │                       │   /ingest (Upload)      │   │
│  └────────┬────────┘                       └────────────┬────────────┘   │
└───────────┼─────────────────────────────────────────────┼────────────────┘
            │                                             │
            ▼                                             ▼
┌──────────────────────┐                    ┌───────────────────────────┐
│  Retrieval API       │                    │  Ingestion API            │
│  (Port 8000)         │                    │  (Port 8001)              │
│  - POST /query       │                    │  - POST /upload           │
│  - GET  /health      │                    │  - GET  /status           │
└──────────┬───────────┘                    └─────────────┬─────────────┘
           │                                              │
           ▼                                              ▼
┌──────────────────────┐                   ┌───────────────────────────────┐
│  Retrieval Pipeline  │                   │  Ingestion Pipeline           │
│                      │                   │  (full corpus rebuild)        │
└──────────┬───────────┘                   └─────────────┬─────────────────┘
           │                                             │
           └─────────────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   Shared Volumes     │
                    │  - data/pdfs/        │
                    │  - storage/          │
                    └──────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Ollama (Port 11434)  │
                    │   or OpenAI API        │
                    └────────────────────────┘
```

### Why This Approach

**Microservices with Shared Storage:**

- **Independent APIs**: Ingestion and Retrieval APIs run as separate services, enabling independent scaling
- **Frontend-Backend Separation**: Next.js frontend communicates with APIs via HTTP, allowing frontend and backend to evolve independently
- **Shared Volume Pattern**: Both services access the same `data/` and `storage/` directories for simplicity
- **CORS Enabled**: APIs allow cross-origin requests from the frontend

**Service Ports:**


| Service              | Port | Purpose                |
| -------------------- | ---- | ---------------------- |
| `frontend`           | 3000 | Next.js UI             |
| `api-retrieval`      | 8000 | Query endpoint         |
| `api-ingestion`      | 8001 | Upload/status endpoint |
| `retrieval` (legacy) | 8501 | (removed) Legacy Streamlit UI |


**Legacy Support:**

- The legacy Streamlit UI (previously served from `app.py` on port 8501) has been removed from the repository; Next.js is the recommended frontend for all current deployments

**Key Alternatives Not Used:**

- **Message Queue (Redis/RabbitMQ)**: Would enable better horizontal scaling but adds operational complexity
- **Separate Databases**: Would decouple services further but increases infrastructure requirements
- **GraphQL**: Overkill for current scope; REST APIs are sufficient

---

## 3. Project Structure

```
backend/
├── src/
│   ├── config.py          # Configuration loading with env var substitution
│   ├── models/            # Data models (Chunk dataclass)
│   ├── pipelines/         # IngestionPipeline & RetrievalPipeline
│   │   ├── base.py        # Factory functions and constants
│   │   ├── ingestion.py   # Document ingestion pipeline
│   │   ├── retrieval.py   # Query retrieval pipeline
│   │   └── utils.py       # File hashing utility
│   ├── stores/            # Vector store (FAISS implementation)
│   ├── loaders/           # Document loader (PDF implementation)
│   ├── splitters/         # Text splitter (Sentence-based)
│   └── adapters/          # LLM & embedding provider abstraction
│       ├── base.py        # Abstract interfaces
│       ├── embedding.py   # OpenAI, Ollama embedders
│       ├── llm.py         # OpenAI, Ollama LLMs
│       ├── nim.py         # NVIDIA NIM adapters
│       └── utils.py       # Shared utilities (connection pooling)
├── api/                   # FastAPI services
│   ├── ingestion/         # Ingestion API (upload + status)
│   │   └── main.py
│   └── retrieval/         # Retrieval API (query)
│       └── main.py
├── app.py                 # (removed) Legacy Streamlit UI was previously here
├── ingest.py              # CLI ingestion tool

frontend/                  # Next.js application
├── app/                  # App router pages
│   ├── page.tsx          # Query page
│   ├── ingest/           # Ingest page
│   │   └── page.tsx
│   └── layout.tsx        # Root layout with navbar
├── src/
│   ├── components/       # React components
│   │   ├── Navbar.tsx
│   │   ├── QueryForm.tsx
│   │   ├── QueryResults.tsx
│   │   ├── FileUploader.tsx
│   │   └── IngestionStatusDisplay.tsx
│   └── lib/
│       └── api.ts        # API client
├── Dockerfile
├── package.json
├── next.config.ts
└── tsconfig.json

tests/                     # Test suite
Dockerfile                 # Multi-purpose Dockerfile
Dockerfile.api             # FastAPI services Dockerfile
docker-compose.yml         # Container orchestration
data/pdfs/                 # PDF upload directory (shared volume)
storage/                   # FAISS index & status files (shared volume)
```

### Module Responsibilities

`**config.py**`

- Loads TOML configuration with `${VAR:-default}` environment variable substitution
- Resolves relative paths against config file location
- Helper functions: `get_storage_dir()`, `get_ingestion_dir()` for consistent path resolution
- Pure configuration—no business logic

`**models/**`

- `Chunk` dataclass for type-safe passage representation (text, source, metadata)

`**pipelines/**`

- High-level workflows coordinating multiple components
- Dependency injection pattern enables testing and customization
- Three ingestion modes: full batch, streaming (batched), incremental (hash-based)

`**stores/`, `loaders/`, `splitters/**`

- Abstract base classes in `base.py` with concrete implementations
- Backward-compatible aliases in `__init__.py` (e.g., `VectorStore = FAISSVectorStore`)

`**adapters/**`

- Provider-agnostic interfaces (`BaseEmbedder`, `BaseLLM`)
- Registry pattern for extensible provider support
- Connection pooling for HTTP-based providers (Ollama)

---

## 4. RAG & LLM Flow

### Ingestion Flow (PDF → Embeddings)

```
User uploads full PDF batch
       │
       ▼
┌──────────────────┐
│IngestionPipeline │
│ run full rebuild │
└──────────────────┘
       │
       ├──► DocumentLoader.load_file()
       │         └──► llama-index SimpleDirectoryReader
       │         └──► List[LlamaDocument]
       │
       ├──► TextSplitter.split_documents()
       │         └──► llama-index SentenceSplitter
       │         └──► List[Chunk] (text, source, metadata)
       │
       ├──► Embedder.embed_batch(texts)
       │         └──► OpenAI: /v1/embeddings
       │         └──► Ollama: /api/embed (batch) or parallel /api/embeddings
       │         └──► List[List[float]] (embeddings)
       │
       └──► VectorStore.add(embeddings, texts, metadatas)
                 └──► FAISS IndexFlatL2 (L2 distance, exact search)
                 └──► metadata.json (text, source, index; NO embeddings)
                 └──► Atomic write with file locking
```

**Key Implementation Detail**: Embeddings are stored ONLY in FAISS, not in metadata JSON. This saves ~50% memory but means FAISS index rebuilds (after source removal) require re-embedding or accepting metadata/FAISS inconsistency.

### Query Flow (Question → Answer)

```
User asks question
       │
       ▼
┌──────────────────┐
│RetrievalPipeline │
│      query()     │
└──────────────────┘
       │
       ├──► Embedder.embed(query)
       │         └──► Same embedding model used at ingestion
       │         └──► List[float] (query embedding)
       │
       ├──► VectorStore.search(query_embedding, k=top_k)
       │         └──► FAISS IndexFlatL2.search()
       │         └──► Returns distances, indices
       │         └──► Lookup metadata by index
       │         └──► List[Dict] (text, source, distance)
       │
       ├──► Context assembly
       │         └──► Join chunk texts with "\n\n"
       │         └──► No token counting (risk of overflow)
       │
       ├──► Prompt construction
       │         └──► Template: "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
       │
       └──► LLM.generate(prompt)
                 └──► OpenAI: chat.completions.create()
                 └──► Ollama: /api/generate with stream=False
                 └──► String response
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGESTION                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PDF ──► SimpleDirectoryReader ──► LlamaDocument[]              │
│                                        │                        │
│                                        ▼                        │
│                               SentenceSplitter                  │
│                                        │                        │
│                                        ▼                        │
│                                    Chunk[]                      │
│                               (text, source, metadata)          │
│                                        │                        │
│                                        ▼                        │
│                              Embedder.embed_batch()             │
│                                        │                        │
│                    ┌───────────────────┴───────────────────┐   │
│                    │                                       │   │
│                    ▼                                       ▼   │
│           FAISS IndexFlatL2                          metadata.json│
│           (embeddings only)                          (text, source)│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ shared storage/
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Query ──► Embedder.embed() ──► Query Vector                    │
│                                        │                        │
│                                        ▼                        │
│                              FAISS.search(k=4)                  │
│                                        │                        │
│                                        ▼                        │
│                              Retrieved Chunks                   │
│                                        │                        │
│                                        ▼                        │
│                              Context Assembly                   │
│                              (joined with "\n\n")               │
│                                        │                        │
│                                        ▼                        │
│                              LLM.generate()                     │
│                                        │                        │
│                                        ▼                        │
│                                   Answer                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Core Components

### 5.1 Document Processing (core.py)

`**DocumentLoader**`

- **Purpose**: Load PDF documents from filesystem
- **Implementation**: Thin wrapper around `llama_index.SimpleDirectoryReader`
- **Input**: Directory path or single file path
- **Output**: `List[LlamaDocument]`
- **Coupling**: Tightly coupled to llama-index for PDF parsing

`**TextSplitter`**

- **Purpose**: Chunk documents while preserving source metadata
- **Implementation**: Wraps `llama_index.SentenceSplitter`
- **Input**: `List[LlamaDocument]`
- **Output**: `List[Chunk]` where `Chunk = (text, source, metadata)`
- **Key Feature**: O(1) source lookup via metadata preservation (avoiding O(n×m) substring matching)

`**VectorStore`**

- **Purpose**: Store and search embeddings with metadata
- **Implementation**: FAISS `IndexFlatL2` with JSON metadata sidecar
- **Storage Format**:
  - `faiss_{model}.index`: Binary FAISS index (embeddings only)
  - `faiss_{model}.json`: Metadata array (text, source, index position)
- **Concurrency**: File locking via `fcntl.flock` for thread/process safety
- **Limitation**: `IndexFlatL2` is O(n) brute-force search—slow at scale (>100k vectors)

### 5.2 Adapters (adapters/)

`**BaseEmbedder` / `BaseLLM`**

- Abstract interfaces decoupling business logic from provider specifics
- Enable testing with mocks and swapping providers without code changes

`**OpenAIEmbedder**`

- Uses `openai.OpenAI` client
- Batch embedding via `client.embeddings.create(input=texts)`
- Dimension mapping hardcoded: `text-embedding-3-small` → 1536, etc.

`**OllamaEmbedder**`

- Primary: `/api/embed` endpoint for batch embedding
- Fallback: Parallel individual requests via `ThreadPoolExecutor` (8 workers) if batch fails
- Connection pooling via `requests.Session`
- Error handling: Tracks failed indices, raises `RuntimeError` with details

`**OpenAILLM**` / `**OllamaLLM**`

- OpenAI: Chat completions API, supports streaming (not currently used)
- Ollama: `/api/generate` or `/api/chat`, **always sets `stream: false`**
- Ollama maps `max_tokens` → `num_predict` option

### 5.3 Pipelines (pipelines.py)

`**IngestionPipeline**`

- **Responsibilities**: Coordinate document loading → splitting → embedding → storage
- **Modes**:
  - `run()`: Load all documents at once (high memory)
  - `run_streaming()`: Batch processing with configurable batch size (default: 100)
- **Corpus Model**: each uploaded batch replaces the previous corpus before a full rebuild
- **Batch Processing**: `_process_file_batch()`, `_embed_and_store()` helpers reduce duplication

`**RetrievalPipeline`**

- **Responsibilities**: Embed query → retrieve context → generate response
- **Components**: Embedder, LLM, VectorStore, context template
- **Context Template**: Hardcoded format (no template engine)
- **No Token Counting**: Risk of exceeding LLM context window

### 5.4 Registry (removed)

---

## 6. Data, State & Configuration

### 6.1 Persistence Model

**Persistent State (Disk):**

- `data/pdfs/*.pdf`: Source documents (user uploads)
- `storage/faiss_{model}.index`: FAISS vector index (binary)
- `storage/faiss_{model}.json`: Metadata sidecar (JSON array)
- `storage/ingestion_status.json`: Status communication between containers

**Computed State (Memory):**

- FAISS index loaded into RAM during operations
- Embeddings stored only in FAISS (not duplicated in metadata)
- Metadata loaded as Python list of dicts


**Ephemeral State:**

- `st.session_state` in the legacy Streamlit UI (pipeline instance caching; UI removed)
- Ingestion pipeline singleton and status in the ingestion API process

### 6.2 Index Lifecycle

**Creation:**

1. `VectorStore.__init__()` creates `IndexFlatL2(dimension)` if no index file exists
2. First `add()` call populates index
3. `save()` persists to disk (called automatically after each `add()`)

**Updates:**

1. Bulk upload replaces the current files in `data/pdfs/`
2. Ingestion clears the active vector index for the selected embedding model
3. All uploaded documents are re-embedded and written back to storage
4. Chat is enabled after the ingestion status becomes `complete`

**Invalidation:**

- `delete_all()`: Resets to empty index
- `--force` flag: Clears index before re-ingestion
- File deletion: Not currently handled (vectors remain orphaned in FAISS)

### 6.3 Configuration System

**TOML + Environment Variable Substitution:**

```toml
[embedding]
provider = "${EMBEDDING_PROVIDER:-openai}"
model = "${EMBEDDING_MODEL:-text-embedding-3-small}"
```

**Resolution Order:**

1. Environment variable value if set
2. Default value after `:-` if variable unset
3. Empty string if no default provided

**Configuration Loading:**

- `load_config(path)` parses TOML
- `_substitute_env_vars()` recursively substitutes `${VAR:-default}` syntax
- `resolve_path()` makes paths relative to config file location

**Secrets Management:**

- `OPENAI_API_KEY` in `.env` file (git-ignored)
- No encryption at rest for API keys
- Keys passed via environment variables to containers

### 6.4 uv Usage

**Dependency Management:**

- `pyproject.toml`: Declares dependencies and dev extras
- `uv.lock`: Locked dependency tree for reproducible builds
- `uv sync`: Install production dependencies
- `uv sync --extra dev`: Install with pytest, ruff, mypy

**Running Commands:**

- `uv run python cli/ingest.py --force`: Run a full ingestion manually
- `uv run pytest`: Execute test suite
- `uv run ruff check .`: Linting
- `uv run mypy backend/`: Type checking

**Why uv:**

- Faster than pip (Rust-based resolver)
- Lock file ensures consistent deployments
- Native Python version management

---

## 7. Key Design Decisions & Tradeoffs

### 7.1 FAISS IndexFlatL2 vs. Approximate Search

**Decision**: Use FAISS `IndexFlatL2` (brute-force exact search)

**Why:**

- Simple to implement and understand
- Exact results (no approximation error)
- No training required

**Tradeoffs:**

- O(n) search complexity—linear slowdown as index grows
- At 100k+ vectors, search becomes noticeably slow
- Memory usage grows linearly with vector count

**Alternative Not Used:** `IndexIVFFlat` or `IndexHNSW` for sublinear search

- Would require training step and hyperparameter tuning
- Adds complexity for current use case

### 7.2 Embeddings Only in FAISS (Not Metadata)

**Decision**: Store embeddings ONLY in FAISS index, not in metadata JSON

**Why:**

- Reduces memory usage by ~50%
- Faster metadata serialization (no large embedding arrays in JSON)
- Metadata JSON stays human-readable

**Tradeoffs:**

- Cannot rebuild FAISS index from metadata alone
- Source removal leaves orphaned vectors in FAISS
- Must re-embed all documents for full consistency

### 7.3 File-Based Status Communication

**Decision**: Use `ingestion_status.json` for container communication

**Why:**

- No additional infrastructure required (no Redis, RabbitMQ)
- Simple to implement and debug
- Works with Docker volumes

**Tradeoffs:**

- Polling-based (not event-driven)
- Potential race conditions (mitigated by atomic writes)
- Not scalable across multiple hosts

### 7.4 Synchronous-Only Architecture

**Decision**: No async/await patterns

**Why:**

- Simpler code (no `asyncio` complexity)
- Easier to debug
- Streamlit is synchronous

**Tradeoffs:**

- Cannot handle concurrent requests efficiently
- Blocking I/O for network calls
- No streaming responses to UI

### 7.5 Provider Registry Pattern

**Decision**: Use registry pattern for LLM/embedder providers

```python
_EMBEDDER_REGISTRY: dict[str, Type[BaseEmbedder]] = {}

def register_embedder(provider: str, cls: Type[BaseEmbedder]):
    _EMBEDDER_REGISTRY[provider] = cls
```

**Why:**

- New providers can be added without modifying existing code
- Clean separation of provider-specific logic
- Runtime provider listing available

**Tradeoffs:**

- Global state (registry is module-level)
- Less explicit than explicit factory functions
- Registration order matters (must import to register)

---

## 8. Reliability, Scalability & Security

### 8.1 Error Handling

**Current Approach:**

- Specific exceptions raised: `FileNotFoundError`, `ValueError`, `RuntimeError`
- Exceptions bubble up to entry points
- Status file captures error messages for UI display
- Logging via standard library `logging` (stdout only)

**Gaps:**

- No retry logic with exponential backoff for transient failures
- No circuit breaker for external API failures
- No graceful degradation (all-or-nothing failure)

**Error Handling in OllamaEmbedder:**

```python
# Good: Tracks failed indices
errors: list[tuple[int, Exception]] = []
# ... populate errors ...
if errors:
    raise RuntimeError(f"Embedding failed for {len(errors)} texts at indices {failed_indices}")
```

### 8.2 Performance Bottlenecks

**Critical Bottlenecks:**

1. **FAISS Search**: O(n) complexity with `IndexFlatL2`. At 100k vectors, search becomes slow.
2. **No Embedding Cache**: Same queries re-embedded every time
3. **Full Metadata Write**: Every `add()` writes entire metadata JSON to disk
4. **UI Blocking**: The removed legacy Streamlit UI previously polled for 2 minutes with `time.sleep(2)`, which blocked the Streamlit thread; the Next.js frontend and FastAPI services do not use that pattern
5. **No Batch Writes**: Individual file writes for each document batch

**File Locking Overhead:**

- Every `VectorStore.add()` acquires exclusive lock
- Mitigates corruption but serializes concurrent writes

### 8.3 Secret Handling

**Current State:**

- API keys in `.env` file (git-ignored)
- Passed to containers via environment variables
- Keys extracted via `kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")`

**Risks:**

- Keys visible in process environment (`ps e`)
- No encryption at rest
- Could be logged if not careful (currently handled via `kwargs.pop`)

**LLM-Specific Risks:**

- No prompt injection protection
- No output filtering
- Context window overflow not checked (could expose unintended data)

### 8.4 Security Gaps

**File Upload (legacy Streamlit UI):**

- Historical issues reported against the removed `app.py` endpoint included: lack of file size limits, missing PDF magic-byte validation, and filename sanitization gaps. These concerns were addressed in the ingestion API (`backend/api/ingestion/main.py`) and in the ingestion pipeline (size limits, magic-byte checks, filename sanitization).

**Input Validation:**

- No query sanitization before sending to LLM
- No rate limiting on Streamlit endpoints

---

## 9. Testing & Quality Overview

This section describes how testing and quality fit into the architecture. **Current gaps, technical debt, and future work are tracked in `docs/ISSUES.md`.**

### 9.1 Test Layout

- Tests live under the `tests/` directory alongside the backend codebase.
- Unit tests focus on core components (`core`, `adapters`, `pipelines`, `stores`, `loaders`, `splitters`).
- Integration and end-to-end tests are expected to exercise ingestion and retrieval flows across services.

### 9.2 How to Run Tests & Checks

- Use `uv` to run tests and checks in a consistent environment:
  - `uv run pytest` – run the full test suite
  - `uv run ruff check .` – lint the codebase
  - `uv run mypy backend/` – type-check the backend

### 9.3 Where to Find Current Issues

For **test coverage gaps**, **known architectural issues**, and **planned improvements**, see the dedicated issues document:

- `docs/ISSUES.md` – canonical source for open issues, technical debt, and future work

---

## Appendix: Configuration Reference

### config.toml

```toml
[embedding]
provider = "${EMBEDDING_PROVIDER:-openai}"      # "openai" or "ollama"
model = "${EMBEDDING_MODEL:-text-embedding-3-small}"
base_url = "${EMBEDDING_BASE_URL:-}"            # e.g., "http://ollama:11434"

[llm]
provider = "${LLM_PROVIDER:-openai}"            # "openai" or "ollama"
model = "${LLM_MODEL:-gpt-4o-mini}"
base_url = "${LLM_BASE_URL:-}"                  # e.g., "http://ollama:11434"

[ingestion]
directory = "data/pdfs"                         # PDF upload directory
chunk_size = 1024                               # Characters per chunk
chunk_overlap = 50                              # Overlap between chunks

[retrieval]
top_k = 4                                       # Number of chunks to retrieve
context_template = """Context information:
{context}

Question: {question}

Answer:"""

[storage]
directory = "storage"                           # FAISS index location
```

### Environment Variables

Required:

- `OPENAI_API_KEY` (if using OpenAI)
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`
- `LLM_PROVIDER`, `LLM_MODEL`

Optional:

- `EMBEDDING_BASE_URL`, `LLM_BASE_URL` (for Ollama)

---

*Document derived from code analysis of RecRAG repository. Last updated: 2026-02-20*
