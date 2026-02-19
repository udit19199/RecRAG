# RecRAG Architecture

A code-driven architecture document for the RecRAG RAG pipeline system.

---

## 1. Overview

### What the System Does

RecRAG is a document ingestion and question-answering system built on the Retrieval-Augmented Generation (RAG) pattern. It processes PDF documents, generates embeddings, stores them in a vector index, and answers user questions using retrieved context.

**Primary Use Cases:**
- **Document Upload & Ingestion**: Users upload PDFs via a web UI. Files are automatically processed into searchable embeddings.
- **Question Answering**: Users ask questions about uploaded documents. The system retrieves relevant passages and generates answers using an LLM.
- **Batch Processing**: CLI tool for one-time ingestion of document collections.

**Non-Goals:**
- Real-time collaborative editing
- Multi-modal content (images, audio, video) - currently PDF/text only
- Production-grade authentication or multi-tenancy
- Automated evaluation or pipeline recommendation (planned but not implemented)

---

## 2. Architecture

### Overall Architectural Style

**Pipeline-based modular monolith with dual-container deployment.**

The system separates concerns into discrete pipelines (ingestion and retrieval) that share state through the filesystem. This is not a microservices architecture—both pipelines run in the same codebase and share libraries, but they can be deployed in separate containers.

```
┌────────────────────────────────────────────────────────────────────┐
│                      Ingestion Pipeline                             │
│  watch.py ──► IngestionPipeline ──► FAISS Index + Metadata         │
│         (file watcher)   (load/split/embed/store)                   │
└────────────────────────────────────────────────────────────────────┘
                               │
                    Shared Volumes (data/, storage/)
                               │
┌────────────────────────────────────────────────────────────────────┐
│                     Retrieval Pipeline                              │
│  app.py ──► RetrievalPipeline ──► Query Response                    │
│   (Streamlit)   (embed/search/generate)                             │
└────────────────────────────────────────────────────────────────────┘
```

### Why This Approach

**Dual-Container Pattern:**
- **Separation of concerns**: Ingestion (background, batch) is fundamentally different from retrieval (interactive, real-time)
- **Resource isolation**: Ingestion can be memory-intensive; separating it prevents UI latency
- **Independent scaling**: Could scale ingestion workers separately from UI instances
- **Simple communication**: File-based status protocol avoids need for message queue or shared database

**Tradeoff**: Shared filesystem couples the containers to the same host/volume. This is simpler than a message queue for the current use case but limits horizontal scaling across hosts.

**Key Alternatives Not Used:**
- **Message Queue (Redis/RabbitMQ)**: Would enable better horizontal scaling but adds operational complexity
- **Single-Process Architecture**: Would simplify deployment but ingestion would block queries
- **Microservices**: Overkill for current scope; shared codebase is simpler to develop and test

---

## 3. Project Structure

```
backend/
├── src/
│   ├── config.py          # Configuration loading with env var substitution
│   ├── core.py            # Document processing & FAISS vector store
│   ├── pipelines.py       # IngestionPipeline & RetrievalPipeline
│   ├── stores/            # Vector store factory (currently FAISS only)
│   ├── loaders/           # Document loader factory (currently PDF only)
│   └── adapters/          # LLM & embedding provider abstraction
│       ├── base.py        # Abstract interfaces
│       ├── embedding.py   # OpenAI & Ollama embedders
│       ├── llm.py         # OpenAI & Ollama LLMs
│       └── utils.py       # Shared utilities (connection pooling)
├── app.py                 # Streamlit UI entry point
├── ingest.py              # CLI ingestion tool
└── watch.py               # File watcher daemon with debouncing

tests/                     # 28 tests covering core and adapters
docker/                    # Container definitions
data/pdfs/                 # PDF upload directory (shared volume)
storage/                   # FAISS index & status files (shared volume)
```

### Module Responsibilities

**`config.py`**
- Loads TOML configuration with `${VAR:-default}` environment variable substitution
- Resolves relative paths against config file location
- Pure configuration—no business logic

**`core.py`**
- Abstract interfaces: `BaseDocumentLoader`, `BaseTextSplitter`, `BaseVectorStore`
- Concrete implementations using llama-index and FAISS
- `Chunk` dataclass for type-safe passage representation
- Thread-safe FAISS operations with file locking

**`pipelines.py`**
- High-level workflows coordinating multiple components
- Dependency injection pattern enables testing and customization
- Three ingestion modes: full batch, streaming (batched), incremental (hash-based)

**`adapters/`**
- Provider-agnostic interfaces (`BaseEmbedder`, `BaseLLM`)
- Registry pattern for extensible provider support
- Connection pooling for HTTP-based providers (Ollama)

---

## 4. RAG & LLM Flow

### Ingestion Flow (PDF → Embeddings)

```
User uploads PDF
       │
       ▼
┌──────────────────┐
│   watch.py       │──► Detects file change (5s debounce)
│  (file watcher)  │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│IngestionPipeline │
│  run_incremental │
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

**`DocumentLoader`**
- **Purpose**: Load PDF documents from filesystem
- **Implementation**: Thin wrapper around `llama_index.SimpleDirectoryReader`
- **Input**: Directory path or single file path
- **Output**: `List[LlamaDocument]`
- **Coupling**: Tightly coupled to llama-index for PDF parsing

**`TextSplitter`**
- **Purpose**: Chunk documents while preserving source metadata
- **Implementation**: Wraps `llama_index.SentenceSplitter`
- **Input**: `List[LlamaDocument]`
- **Output**: `List[Chunk]` where `Chunk = (text, source, metadata)`
- **Key Feature**: O(1) source lookup via metadata preservation (avoiding O(n×m) substring matching)

**`VectorStore`**
- **Purpose**: Store and search embeddings with metadata
- **Implementation**: FAISS `IndexFlatL2` with JSON metadata sidecar
- **Storage Format**:
  - `faiss_{model}.index`: Binary FAISS index (embeddings only)
  - `faiss_{model}.json`: Metadata array (text, source, index position)
- **Concurrency**: File locking via `fcntl.flock` for thread/process safety
- **Limitation**: `IndexFlatL2` is O(n) brute-force search—slow at scale (>100k vectors)

### 5.2 Adapters (adapters/)

**`BaseEmbedder` / `BaseLLM`**
- Abstract interfaces decoupling business logic from provider specifics
- Enable testing with mocks and swapping providers without code changes

**`OpenAIEmbedder`**
- Uses `openai.OpenAI` client
- Batch embedding via `client.embeddings.create(input=texts)`
- Dimension mapping hardcoded: `text-embedding-3-small` → 1536, etc.

**`OllamaEmbedder`**
- Primary: `/api/embed` endpoint for batch embedding
- Fallback: Parallel individual requests via `ThreadPoolExecutor` (8 workers) if batch fails
- Connection pooling via `requests.Session`
- Error handling: Tracks failed indices, raises `RuntimeError` with details

**`OpenAILLM`** / **`OllamaLLM`**
- OpenAI: Chat completions API, supports streaming (not currently used)
- Ollama: `/api/generate` or `/api/chat`, **always sets `stream: false`**
- Ollama maps `max_tokens` → `num_predict` option

### 5.3 Pipelines (pipelines.py)

**`IngestionPipeline`**
- **Responsibilities**: Coordinate document loading → splitting → embedding → storage
- **Modes**:
  - `run()`: Load all documents at once (high memory)
  - `run_streaming()`: Batch processing with configurable batch size (default: 100)
  - `run_incremental()`: MD5 hash-based change detection, only processes new/changed files
- **State Tracking**: `processed_files.json` maps file paths to MD5 hashes
- **Batch Processing**: `_process_file_batch()`, `_embed_and_store()` helpers reduce duplication

**`RetrievalPipeline`**
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
- `storage/processed_files.json`: File hash tracking for incremental ingestion
- `storage/ingestion_status.json`: Status communication between containers

**Computed State (Memory):**
- FAISS index loaded into RAM during operations
- Embeddings stored only in FAISS (not duplicated in metadata)
- Metadata loaded as Python list of dicts

**Ephemeral State:**
- `st.session_state` in Streamlit (pipeline instance caching)
- Thread-local state in file watcher (debounce timers, pending files)

### 6.2 Index Lifecycle

**Creation:**
1. `VectorStore.__init__()` creates `IndexFlatL2(dimension)` if no index file exists
2. First `add()` call populates index
3. `save()` persists to disk (called automatically after each `add()`)

**Updates:**
1. Incremental ingestion checks MD5 hashes in `processed_files.json`
2. Changed files trigger `_remove_by_sources()` (removes metadata only)
3. New embeddings appended to FAISS index
4. **Limitation**: FAISS index not rebuilt; old vectors remain (metadata inconsistency)

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
- `uv run python backend/watch.py`: Run in virtual environment
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
4. **UI Blocking**: `app.py` polls for 2 minutes with `time.sleep(2)` blocking the Streamlit thread
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

**File Upload (app.py:80-96):**
- No file size limits (memory exhaustion risk)
- No PDF magic byte validation (`%PDF-`)
- No filename sanitization (path traversal risk via `../`)

**Input Validation:**
- No query sanitization before sending to LLM
- No rate limiting on Streamlit endpoints

---

## 9. Testing, Technical Debt & Future Work

### 9.1 Test Coverage

**Current State (28 tests):**
- `test_core.py`: 11 tests covering Chunk, DocumentLoader, TextSplitter, VectorStore
- `test_adapters/test_embedding.py`: 6 tests for OpenAI and Ollama embedders
- `test_adapters/test_llm.py`: 11 tests for OpenAI and Ollama LLMs

**Coverage Gaps:**
- No integration tests (end-to-end flow)
- No tests for `watch.py` file watcher
- No tests for `pipelines.py` (only unit tests for components)
- No tests for `config.py` edge cases
- No tests for error recovery paths

**Test Quality:**
- Good use of fixtures in `conftest.py`
- Mocking external APIs properly
- Missing: Property-based tests, load tests

### 9.2 Known Architectural Issues

**Critical:**
1. **FAISS/Metadata Inconsistency**: Source removal updates metadata but not FAISS index
2. **No Document Deletion**: Can add documents, cannot remove them cleanly
3. **Context Window Risk**: No token counting before LLM calls

**High Priority:**
4. **UI Blocking**: Streamlit thread blocked during status polling
5. **No Async**: Cannot handle concurrent requests
6. **Scalability Ceiling**: `IndexFlatL2` won't scale beyond ~100k vectors

**Medium Priority:**
7. **No Caching**: Repeated queries/documents re-embedded
8. **No Metrics**: No visibility into performance or quality
9. **No Validation**: Configuration values not validated at startup

### 9.3 Practical Improvements (Codebase-Aligned)

**Immediate (Low Effort):**
1. Add file size validation in `app.py` (check `uploaded_file.size`)
2. Add PDF magic byte validation (`%PDF-` header check)
3. Implement simple in-memory LRU cache for embeddings
4. Add token estimation before LLM calls (tiktoken for OpenAI)

**Short Term (Medium Effort):**
5. Replace `IndexFlatL2` with `IndexIVFFlat` for better scalability
6. Add async support using `asyncio` and `aiohttp` for Ollama
7. Implement proper document deletion with FAISS ID mapping
8. Add metrics collection (latency, token counts, cache hit rates)

**Long Term (High Effort):**
9. Implement evaluation framework (answer relevance, retrieval accuracy)
10. Add hybrid search (vector + keyword BM25)
11. Support multi-modal content (images via vision models)
12. Implement recommendation engine for pipeline configuration

### 9.4 Inconsistencies & Ambiguities

**Unresolved Questions:**
1. **Factory Modules**: `stores/__init__.py` and `loaders/__init__.py` only support single provider—should they be expanded or removed?
2. **Streaming**: Interface supports streaming (`supports_streaming` property) but never used—intentional or oversight?

**Code Smells:**
1. **Large Pipeline Class**: `IngestionPipeline` has multiple responsibilities (discovery, hashing, batching, embedding)
2. **Magic Strings**: File extensions (`.pdf`), config keys scattered throughout code

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

*Document derived from code analysis of RecRAG repository. Last updated: 2026-02-19*
