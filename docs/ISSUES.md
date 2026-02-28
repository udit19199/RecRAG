# RecRAG - Outstanding Issues

Issues identified during code review, grouped by severity and category.

---

## Critical Priority

### C1. `get_vector_store_paths` ImportError — Service Cannot Start

**File**: `backend/src/pipelines/__init__.py:8`, `backend/src/pipelines/base.py`

`get_vector_store_paths` is imported and re-exported in `pipelines/__init__.py` but the function does not exist anywhere in `pipelines/base.py`. It is a stale leftover from the FAISS era.

**Impact**: `ImportError` on any import of the `pipelines` package. Every execution path (API, Streamlit app, CLI ingest, watcher) crashes at startup.

**Solution**: Remove the stale symbol from `pipelines/__init__.py` and from the `from .base import (...)` block. Verify with `python -c "from pipelines import get_retrieval_pipeline"`.

---

### C2. `file.filename` Is `None`-Unguarded in Upload Endpoint

**File**: `backend/api/ingestion/main.py:93` — `upload_pdf()`

FastAPI's `UploadFile.filename` is typed `str | None`. Both `.lower()` and `Path(file.filename).name` raise `AttributeError`/`TypeError` when no filename header is sent.

**Impact**: Any automated client (curl, test harness) that omits the filename triggers an unhandled 500 instead of a clean 400.

**Solution**:
```python
if not file.filename:
    raise HTTPException(status_code=400, detail="Filename is required")
if not file.filename.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Only PDF files are allowed")
```

---

### C3. Blocking Filesystem I/O Inside `async` Upload Route

**File**: `backend/api/ingestion/main.py:109` — `upload_pdf()`

```python
content = await file.read()
with open(file_path, "wb") as f:
    f.write(content)   # synchronous, blocks the event loop
```

`f.write(content)` is a synchronous call inside an `async def` route handler. FastAPI runs on an asyncio event loop; this blocks all other coroutines for the full duration of the write (up to 50 MB).

**Impact**: Under any real load, all concurrent requests queue behind each upload. A single large upload stalls the entire service.

**Solution**: Use `aiofiles`:
```python
import aiofiles

async with aiofiles.open(file_path, "wb") as f:
    await f.write(content)
```

---

## High Priority

### 1. Hardcoded Timeouts

**Files**: `adapters/embedding.py:78,109`, `adapters/llm.py:99,111`

- 30s for embeddings
- 120s for LLM

Not configurable. No retry logic.

**Solution**: Add to `config.toml`:
```toml
[adapters]
timeout_embedding = 30
timeout_llm = 120
max_retries = 3
```

---

### 2. Streamlit Blocking Poll

**File**: `backend/app.py:122-136`

```python
for _ in range(60):
    time.sleep(2)  # Blocks UI thread for 2 minutes!
```

**Impact**: UI frozen, poor user experience.

**Solution**: Use `st.rerun()` with `st.session_state` for non-blocking updates.

Architecture context: see `docs/ARCHITECTURE.md` §8.2 **Performance Bottlenecks** (UI blocking) and §2 **Architecture** (Streamlit vs API services).

---

### 3. Non-Atomic Status File Writes

**File**: `backend/watch.py:74-75`

```python
with open(self.status_file, "w") as f:
    json.dump(status_data, f, indent=2)
```

Process crash mid-write corrupts the file.

**Solution**: Write to temp file, then atomic rename.

Architecture context: see `docs/ARCHITECTURE.md` §7.3 **File-Based Status Communication** and §6.1 **Persistence Model**.

---

### 4. `NIMEmbedder` Makes a Live API Call at Construction Time

**File**: `backend/src/adapters/nim.py:44` — `NIMEmbedder.__init__()`

`_detect_dimension()` makes a real NVIDIA NIM API call (`get_query_embedding("test")`) every time an embedder is instantiated — including during pipeline initialization and in tests.

**Impact**: Application startup fails on transient NIM outages. Tests require real API keys or complex constructor-level mocking. Every pipeline restart burns an unnecessary API call.

**Solution**: Accept `dimension` as an explicit parameter; perform detection lazily only when the value is actually needed:
```python
def __init__(self, model: str, dimension: int | None = None, ...):
    self._dimension: int | None = dimension

@property
def dimension(self) -> int:
    if self._dimension is None:
        self._dimension = len(self._client.get_query_embedding("test"))
    return self._dimension
```

---

### 5. `get_pipeline()` Race Condition in Retrieval API

**File**: `backend/api/retrieval/main.py:78` — `get_pipeline()`

```python
if _pipeline is None:
    _pipeline = get_retrieval_pipeline(...)  # threads A and B both see None
```

Two concurrent requests can both observe `_pipeline is None` and simultaneously initialize it, opening duplicate Milvus connections and LLM clients.

**Impact**: Duplicate resource allocation; potential for inconsistent state under concurrent startup traffic.

**Solution**: Double-checked locking:
```python
import threading
_pipeline_lock = threading.Lock()

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = get_retrieval_pipeline(find_config_path())
    return _pipeline
```

---

### 6. HTTP Retry Strategy Retries Non-Idempotent POSTs

**File**: `backend/src/adapters/utils.py:21` — `create_session_with_pooling()`

`max_retries=3` is passed as a plain integer. `requests` converts this to `Retry(total=3)`, which retries on **all HTTP methods including POST**. Both embedding and generation endpoints use POST.

**Impact**: A transient 429 or 503 from the Ollama embedding endpoint causes the same batch to be resubmitted, generating duplicate vectors in the store. Corrupts incremental ingestion state.

**Solution**: Restrict retries to safe methods only:
```python
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    allowed_methods={"GET"},   # never retry POSTs
    backoff_factor=0.5,
    status_forcelist={429, 502, 503, 504},
)
```

---

### 7. Each File Is Hashed Twice per Ingestion Run

**File**: `backend/src/pipelines/ingestion.py` — `_get_changed_files()` + `_process_files_in_batches()`

`_compute_file_hash(file_path)` is called once in `_get_changed_files()` to detect changes, and again in `_process_files_in_batches()` to record the final hash. For large PDF corpora this doubles disk I/O.

**Impact**: For 100 × 50 MB PDFs, ~5 GB of redundant reads per run. Measurable regression at scale.

**Solution**: Return the computed hash from `_get_changed_files` and thread it through to `_process_files_in_batches`:
```python
def _get_changed_files(...) -> list[tuple[Path, bool, str]]:  # add str hash
    ...
    return [(path, is_new, file_hash), ...]
```

---

### 8. `pending_files` Race Condition Silently Drops Files

**File**: `backend/watch.py` — `_trigger_ingestion()` / `_run_ingestion_with_lock()`

The lock is released between clearing `pending_files` and checking `is_processing`. A file event arriving in that window is added to `pending_files`, starts a debounce timer, but when the timer fires it finds `is_processing=True` and returns immediately — the file is now lost from `pending_files` with no processing queued.

**Impact**: Files uploaded during an active ingestion run are silently skipped until the user uploads another file to re-trigger the watcher.

**Solution**: Don't clear `pending_files` in `_trigger_ingestion` if `is_processing` is already True. Let the debounce mechanism naturally queue the next batch after the current run completes.

---

### 9. Wildcard CORS Allows Any Origin in Production

**Files**: `backend/api/ingestion/main.py:64`, `backend/api/retrieval/main.py:58`

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

**Impact**: In production, any origin on the internet can make cross-origin requests to these APIs, enabling CSRF-style attacks.

**Solution**: Drive allowed origins from an environment variable:
```python
import os
origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, ...)
```

---

## Medium Priority

### 4. No Embedding Cache

Same query re-embedded every time. No caching at any level.

**Impact**: Wasted API calls, slower responses.

**Solution**: Add LRU cache with configurable size (default 1000 entries).

Architecture context: see `docs/ARCHITECTURE.md` §7.2 **Embeddings Only in FAISS (Not Metadata)** and §8.2 **Performance Bottlenecks**.

---

### 5. Full JSON Write on Every Add

**File**: `backend/src/stores/faiss.py:58`

Every `add()` writes entire metadata JSON to disk. For bulk ingestion of 10k documents, that's 10k full file writes.

**Solution**: Batch writes with explicit `flush()` or auto-flush at batch limit.

---

### 6. IndexFlatL2 Doesn't Scale

**File**: `backend/src/stores/faiss.py:29`

`IndexFlatL2` is brute-force O(n) search. At 100k+ vectors, search becomes noticeably slow.

**Solution**: Use `IndexIVFFlat` for approximate search at scale.

Architecture context: see `docs/ARCHITECTURE.md` §7.1 **FAISS IndexFlatL2 vs. Approximate Search** and §8.2 **Performance Bottlenecks**.

---

### 7. `tiktoken` Used for Token Counting with Non-OpenAI Models

**File**: `backend/src/pipelines/retrieval.py:112` — `generate()`

```python
model = getattr(self.llm, "model", "gpt-4")
```

When the LLM is Ollama or NIM, `tiktoken.encoding_for_model()` silently falls back to `cl100k_base` (the GPT-4 tokenizer). Llama-based models use a different vocabulary; the token count error can be ±30%.

**Impact**: Context is either over-truncated (worse answer quality) or exceeds the model's actual context window (runtime error or silent truncation by the backend).

**Solution**: Expose a `tokenizer_model` config key; default explicitly to `cl100k_base` for non-OpenAI models and document the approximation in config comments.

---

### 8. API Key Persisted in `self.kwargs` After Construction

**Files**: `backend/src/adapters/embedding.py:26` — `OpenAIEmbedder.__init__()`, `backend/src/adapters/llm.py` — `OpenAILLM.__init__()`

`kwargs.pop("api_key")` is called **after** `super().__init__(model, **kwargs)`.  `super().__init__` stores `self.kwargs = kwargs`, but Python's `**` expansion already created a separate copy at the call site — so `self.kwargs` still holds `api_key` even after the pop.

**Impact**: API keys persist on the object. Any repr, logging, or serialization of the adapter leaks credentials.

**Solution**: Extract sensitive kwargs **before** calling `super()`:
```python
def __init__(self, model: str = "...", **kwargs: Any):
    api_key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
    base_url = kwargs.pop("base_url", None)
    super().__init__(model, **kwargs)   # self.kwargs no longer contains secrets
```

---

### 9. `FAISSVectorStore` Is Dead Code but Still Exported

**Files**: `backend/src/stores/faiss.py`, `backend/src/stores/__init__.py`

`FAISSVectorStore` is a complete 200-line implementation but `stores/__init__.py` aliases `VectorStore = MilvusVectorStore`. FAISS is never instantiated by any pipeline. Additionally, `fcntl` locking inside the class is UNIX-only and would break on Windows.

**Impact**: Dead code receives no maintenance — bugs and API drift accumulate silently. Contributors may spend time on it believing it is a supported backend.

**Solution**: Remove `faiss.py` and drop `faiss` from dependencies, or move it to a `contrib/legacy/` namespace with a deprecation notice. Document the migration in `ARCHITECTURE.md`.

---

### 10. `validate_file()` Reads Entire File Into Memory for Magic-Byte Check

**File**: `backend/app.py:56` — `validate_file()`

```python
content = uploaded_file.getvalue()   # loads entire file
if not content.startswith(PDF_MAGIC_BYTES):
```

Only the first 5 bytes are inspected, but the full file (up to 50 MB) is allocated in memory.

**Impact**: 50 MB allocation per validation call, before the file is even saved. Multiplied across concurrent uploads, this is a memory pressure vector.

**Solution**:
```python
header = uploaded_file.getvalue()[:5]
if header != PDF_MAGIC_BYTES:
    return False, "Invalid PDF: missing PDF header"
```

---

### 11. `get_evaluator()` Creates a New Client Instance on Every Call

**Files**: `backend/src/evaluation/ragas_eval.py:98`, `backend/app.py:157`

`get_evaluator()` instantiates a new `RagasEvaluator` — including a new `OpenAI` client and `llm_factory` call — on every evaluation request from the Streamlit UI.

**Impact**: Unnecessary object churn and latency on every scored query. `llm_factory` initialization is non-trivial.

**Solution**: Cache as a module-level singleton:
```python
_evaluator: RagasEvaluator | None = None

def get_evaluator() -> RagasEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RagasEvaluator()
    return _evaluator
```

---

## Low Priority

### 7. No Typed Config Class

Config passed as `dict[str, Any]` everywhere. No IDE autocomplete, no validation.

**Solution**: Pydantic model with validation.

---

### 8. No Circuit Breaker

If OpenAI/Ollama is down, entire system fails. No fallback, no graceful degradation.

---

### 9. No Document Deletion API

Can add documents but cannot remove them from the UI.

Architecture context: see `docs/ARCHITECTURE.md` §6.2 **Index Lifecycle** and §7.2 **Embeddings Only in FAISS (Not Metadata)**.

---

### 10. Logging Only to Stdout

No file logging, no structured logging, no log levels in config.

---

### 11. MD5 Used for File Change Detection

**File**: `backend/src/pipelines/utils.py:6` — `_compute_file_hash()`

MD5 is a broken hash function. While not a direct security vulnerability in this context (the hash is only used for change detection, not authentication), its use raises questions in security reviews and it is algorithmically slower than modern alternatives for large binary files.

**Solution**: Replace with `hashlib.sha256()` (no extra dependency) without changing callers.

---

### 12. `dimension` vs `dimensions` Kwarg Inconsistency Across Embedders

**Files**: `backend/src/adapters/embedding.py:59` (`OpenAIEmbedder`) vs `backend/src/adapters/embedding.py:83` (`OllamaEmbedder`)

`OpenAIEmbedder` reads a custom dimension via `kwargs.get("dimensions")` (plural), while `OllamaEmbedder` uses `kwargs.get("dimension")` (singular). Both silently ignore the wrong spelling.

**Impact**: Passing `dimension=1024` to `OpenAIEmbedder` is silently ignored; passing `dimensions=1024` to `OllamaEmbedder` is silently ignored. No error is raised.

**Solution**: Standardize on `dimension` (singular) across all adapters. Add a validation error for unrecognized dimension-related kwargs.

---

### 13. Adapter Registry Has No Thread Safety

**File**: `backend/src/adapters/__init__.py`

The `_EMBEDDER_REGISTRY` and `_LLM_REGISTRY` dicts are populated via module-level side effects at import time. The pattern is valid under CPython's GIL for the initial load, but makes the registry hard to test in isolation and fragile if registration logic ever becomes conditional.

**Solution**: Move registrations into an explicit `_register_defaults()` function, or use `__init_subclass__` on `BaseEmbedder`/`BaseLLM` for auto-registration.

---

## Security Considerations

| Risk | Location | Status |
|------|----------|--------|
| File upload DoS | `app.py` | ✅ Fixed (size limits) |
| Path traversal | `app.py` | ✅ Fixed (filename sanitization) |
| Malformed PDFs | `app.py` | ✅ Fixed (magic byte check) |
| Prompt injection | `retrieval.py` | Open (complex to mitigate) |
| API key exposure | `.env` | Use secrets manager in production |
| API key leak in `self.kwargs` | `adapters/embedding.py`, `adapters/llm.py` | Open — see Medium Priority issue 8 |
| Wildcard CORS in production | `api/ingestion/main.py`, `api/retrieval/main.py` | Open — see High Priority issue 9 |
| Unauthenticated upload endpoint | `api/ingestion/main.py` | Open — any caller can upload files |

---

## Architectural Issues & Future Work

### Test Coverage

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
- No smoke test confirming all top-level packages import without error (would have caught C1 immediately)
- No tests for `NIMEmbedder` dimension detection or API failure handling
- No tests for `get_pipeline()` concurrent initialization

**Test Coverage Summary:**

| Component | Status | Notes |
|-----------|--------|-------|
| Core (VectorStore, Splitter, Loader) | Tested | 11 tests |
| Adapters (OpenAI, Ollama) | Tested | 17 tests with mocks |
| Pipelines | Missing | No tests |
| Watch daemon | Missing | No tests |
| Config edge cases | Missing | No tests |
| Integration (E2E) | Missing | No tests |

**Test Quality:**

- Good use of fixtures in `conftest.py`
- Mocking external APIs properly
- Missing: Property-based tests, load tests

---

### Known Architectural Issues

**Critical:**

1. **No Document Deletion**: Can add documents but cannot remove them cleanly. See Low Priority issue 9 **“No Document Deletion API”** above.
2. **Context Window Risk**: No token counting before LLM calls.3. **Startup ImportError**: `get_vector_store_paths` export breaks all entrypoints. See Critical issue C1.
4. **Blocking Async I/O**: Upload endpoint blocks event loop during file write. See Critical issue C3.
**High Priority:**

3. **UI Blocking**: Streamlit thread blocked during status polling. See High Priority issue 2 **“Streamlit Blocking Poll”**.
4. **No Async**: Cannot handle concurrent requests.
5. **Scalability Ceiling**: `IndexFlatL2` won't scale beyond ~100k vectors. See Medium Priority issue 6 **“IndexFlatL2 Doesn't Scale”**.

**Medium Priority:**

6. **No Caching**: Repeated queries/documents re-embedded. See Medium Priority issue 4 **“No Embedding Cache”**.
7. **No Metrics**: No visibility into performance or quality.
8. **No Validation**: Configuration values not validated at startup.

Architecture context:

- Context window handling and prompt construction: see `docs/ARCHITECTURE.md` §4 **RAG & LLM Flow** (Query Flow) and §7.2 **Embeddings Only in FAISS (Not Metadata)**.
- Synchronous design and scalability limits: see `docs/ARCHITECTURE.md` §7.4 **Synchronous-Only Architecture** and §8.2 **Performance Bottlenecks**.

---

### Practical Improvements

**Immediate (Low Effort):**

1. Implement simple in-memory LRU cache for embeddings.
2. Add token estimation before LLM calls (tiktoken for OpenAI).

**Short Term (Medium Effort):**

5. Replace `IndexFlatL2` with `IndexIVFFlat` for better scalability.
6. Add async support using `asyncio` and `aiohttp` for Ollama.
7. Implement proper document deletion with FAISS ID mapping.
8. Add metrics collection (latency, token counts, cache hit rates).

**Long Term (High Effort):**

9. Implement evaluation framework (answer relevance, retrieval accuracy).
10. Add hybrid search (vector + keyword BM25).
11. Support multi-modal content (images via vision models).
12. Implement recommendation engine for pipeline configuration.

---

### Inconsistencies & Open Questions

**Unresolved Questions:**

1. **Factory Modules**: `stores/__init__.py` and `loaders/__init__.py` only support single provider—should they be expanded or removed?
2. **Streaming**: Interface supports streaming (`supports_streaming` property) but never used—intentional or oversight?

**Code Smells:**

1. **Large Pipeline Class**: `IngestionPipeline` has multiple responsibilities (discovery, hashing, batching, embedding).
2. **Magic Strings**: File extensions (`.pdf`), config keys scattered throughout code.

**Resolved (2026-03-01, code review):**

- Open: API keys retained in `self.kwargs` after construction — see Medium Priority issue 8.
- Open: `tiktoken` used for all LLM providers regardless of tokenizer compatibility — see Medium Priority issue 7.
- Open: `FAISSVectorStore` is unreferenced dead code — see Medium Priority issue 9.

**Resolved (2026-02-20):**

- ~~`core.py` facade~~: Removed deprecated re-export module.
- ~~Path resolution duplication~~: Added `get_storage_dir()`, `get_ingestion_dir()` helpers.
- ~~`sys.path.insert()` hacks~~: Replaced with proper package installation via `pyproject.toml`.
- ~~Private attribute inconsistency~~: Standardized to `_dimension` across embedders.

---

### Performance Notes

1. **Large corpora**: Current implementation loads all documents into memory during full ingestion. Use `process_documents_streaming()` for lower memory usage.

2. **Concurrent requests**: No request queueing. High load could overwhelm the system.

3. **Index rebuilds**: ✅ Fixed - Source removal now rebuilds FAISS index to remove orphaned vectors.
