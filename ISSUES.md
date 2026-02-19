# RecRAG - Outstanding Issues

This document tracks outstanding code quality, performance, and architecture issues identified during code review.

---

## 🔴 Critical

### 1. Tests Added ✅

Test suite now exists with comprehensive coverage:
- `tests/test_core.py` - Core component tests
- `tests/test_adapters/test_embedding.py` - Embedding adapter tests
- `tests/test_adapters/test_llm.py` - LLM adapter tests

**Status:** Resolved in recent refactoring.

---

### 2. Data Loss in Parallel Embedding - Fixed ✅

**File:** `adapters/embedding.py:142-171`

The `_embed_batch_parallel()` method now properly handles failures:
- Tracks errors with index information
- Raises descriptive `RuntimeError` with failed indices
- No silent dropping of failed embeddings

**Status:** Fixed in recent refactoring.

---

### 3. Dead Code in `_remove_by_sources()` - Fixed ✅

**File:** `core.py:236-259`

The `_remove_by_sources()` method was cleaned up:
- Removed misleading dead code that suggested FAISS index rebuilding
- Method now honestly only removes metadata entries
- Documented that full FAISS index consistency requires re-indexing

**Status:** Fixed in recent refactoring.

---

### 4. Critical Indentation Bug in watch.py - Fixed ✅

**File:** `watch.py:130-181`

The `run_ingestion_safe` function was incorrectly defined outside the class due to indentation error. This has been:
- Fixed indentation to make it a proper method (`_run_ingestion_with_lock`)
- Renamed for clarity
- Verified to work correctly

**Status:** Fixed in recent refactoring.

---

## 🟠 High Priority

### 4. Placeholder Project Metadata - Fixed ✅

**File:** `pyproject.toml`

Project metadata has been updated with proper values.

**Status:** Fixed in recent refactoring.

---

### 5. Dev Dependencies Added ✅

**File:** `pyproject.toml`

Dev dependencies are now properly configured:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
```

Install with: `uv sync --extra dev`

**Status:** Fixed in recent refactoring.

---

### 6. Hardcoded Timeouts

**Files:** `embedding.py:87`, `llm.py:93,114`

- 30s for embeddings, 120s for LLM
- Not configurable via config
- No retry logic with exponential backoff

**Recommended Actions:**
- Add timeout settings to `config.toml`
- Implement retry logic with configurable max retries
- Add circuit breaker pattern for external API calls

---

### 7. Streamlit Polling Anti-Pattern

**File:** `app.py:105-119`

```python
for _ in range(60):
    time.sleep(2)  # Blocking for 2 minutes!
```

This blocks the UI thread. Should use `st.rerun()` or WebSockets.

**Recommended Actions:**
- Use `st.rerun()` with `st.session_state` for status updates
- Consider Server-Sent Events (SSE) or WebSockets for real-time updates
- Add a manual refresh button as fallback

---

## 🟡 Medium Priority

### 8. No File Upload Validation

**File:** `app.py:80-96`

Current implementation has no validation:
- No file size limits (memory exhaustion risk)
- No PDF header validation (could be malicious/non-PDF)
- No filename sanitization (potential path traversal)

**Recommended Actions:**
- Add configurable max file size (e.g., 50MB)
- Validate PDF magic bytes (`%PDF-`)
- Sanitize filenames to remove path separators

---

### 9. Context Window Overflow

**File:** `pipelines.py:218-219`

```python
context_text = "\n\n".join([doc.get("text", "") for doc in context])
```

No check if concatenated context exceeds LLM's context window. Could cause:
- API errors from OpenAI/Ollama
- Truncated responses
- Wasted tokens

**Recommended Actions:**
- Estimate token count before sending
- Truncate or summarize context if too long
- Make context window size configurable

---

### 10. Missing Type Annotations

**File:** `core.py:46`

```python
def split_documents(self, documents: list) -> list[str]:  # list of what?
```

Should be `list[LlamaDocument]` or `list[Document]`.

**Recommended Actions:**
- Add complete type annotations to all public methods
- Run mypy in CI pipeline
- Add py.typed marker for package distribution

---

### 11. FAISS IndexFlatL2 Doesn't Scale

`IndexFlatL2` is brute-force search. At scale (>100k vectors), this becomes very slow.

**Recommended Actions:**
- Implement `IndexIVFFlat` or `IndexHNSW` for production workloads
- Add configuration option to select index type
- Implement index parameter tuning based on dataset size

---

### 12. No Embedding Caching

Re-embedding the same query/document has no caching. Could use simple LRU cache or persistent cache.

**Recommended Actions:**
- Add in-memory LRU cache with configurable size
- Consider persistent cache (e.g., Redis, SQLite) for production
- Add cache hit/miss metrics for monitoring

---

### 13. Full JSON Write on Every Add

**File:** `core.py` (VectorStore)

Every `add()` writes the entire metadata JSON to disk. For bulk ingestion, should batch writes.

**Recommended Actions:**
- Add `flush()` method for explicit persistence
- Implement batched writes with configurable batch size
- Consider append-only log format for better performance

---

### 14. Connection Pooling Added ✅

**Files:** `adapters/embedding.py`, `adapters/llm.py`, `adapters/utils.py`

Connection pooling has been implemented:
- New `adapters/utils.py` with `create_session_with_pooling()` helper
- Both OllamaEmbedder and OllamaLLM use pooled sessions
- Reduces connection overhead for batch operations

**Status:** Fixed in recent refactoring.

---

### 15. No Abstract Config Class

Config is passed as raw `dict[str, Any]` everywhere. Should have typed `Config` dataclass/pydantic model for:
- IDE autocomplete
- Validation at startup
- Clear documentation

**Recommended Actions:**
- Create `Config` dataclass with nested sections
- Use pydantic for validation
- Add config schema documentation

---

### 16. No Graceful Degradation

If Ollama/OpenAI is down, the entire system fails. No circuit breaker, no fallback.

**Recommended Actions:**
- Implement circuit breaker pattern
- Add fallback providers
- Return meaningful error messages to users

---

### 17. Status File Protocol is Fragile

`ingestion_status.json` can be corrupted if process dies mid-write. Should use atomic writes (write to temp, then rename).

**Recommended Actions:**
- Implement atomic writes using temp file + rename
- Add status file validation on startup
- Consider using SQLite for status tracking

---

## 📋 Missing Features

| Feature | Impact | Notes |
|---------|--------|-------|
| Unit tests | Critical | No test coverage exists |
| Integration tests | Critical | No E2E validation |
| Type checking (mypy) | High | Referenced but not configured |
| Logging to file | Medium | Only stdout |
| Metrics/Monitoring | Medium | No observability |
| Rate limiting | Medium | Could hit API limits |
| Document deletion | Medium | Can only add, never remove |
| Incremental ingestion | Medium | Always re-ingests all |
| API versioning | Low | No version on status file |

---

## ⚡ Performance Considerations

1. **Large Document Sets**: Current implementation loads all documents into memory. Consider streaming/chunked processing for very large datasets.

2. **Concurrent Requests**: No request queueing or throttling. High load could overwhelm the system.

3. **Index Rebuilding**: `_remove_by_sources()` rebuilds the entire FAISS index. Consider more efficient approaches for incremental updates.

---

## 🔒 Security Considerations

1. **Input Validation**: No validation on uploaded PDFs (size, format, content). Malformed files could cause memory issues or crashes.

2. **Path Traversal**: File paths should be validated to prevent directory traversal attacks when saving uploads.

3. **Rate Limiting**: No protection against DoS attacks on the Streamlit endpoint.

4. **File Upload**: No virus scanning or content validation for uploaded files.

---

## ✅ Resolved Issues

The following issues were fixed during code review and optimization:

### Initial Code Review
- [x] Duplicate Metadata on Re-ingestion (file locking + source-based removal)
- [x] Ollama Batch Embedding Sequential (parallel with ThreadPoolExecutor)
- [x] Race Condition in VectorStore (file locking added)
- [x] API Key Exposure Risk (using kwargs.pop)
- [x] Inconsistent Import Paths (standardized sys.path.insert)
- [x] Redundant Embedder Initialization (extracted helper functions)
- [x] Dead Code in pipelines.py (redundant list comprehension removed)
- [x] Tight Coupling in Pipelines (refactored with helper functions)

### Performance Optimization
- [x] O(n×m) Chunk-to-Source Mapping (refactored to O(n) using Chunk dataclass)
- [x] No Debouncing in File Watcher (5-second debounce timer implemented)

### Major Refactoring (Latest)
- [x] Critical bug: Fixed `run_ingestion_safe` indentation in watch.py
- [x] Added comprehensive test suite (28 tests across core and adapters)
- [x] Fixed OllamaEmbedder error handling (no more silent failures)
- [x] Added connection pooling for Ollama adapters
- [x] Extracted shared utilities (`adapters/utils.py`)
- [x] Consolidated factory functions in pipelines.py
- [x] Extracted batch processing logic to reduce duplication
- [x] Added named constants throughout (DEFAULT_BATCH_SIZE, etc.)
- [x] Simplified adapter implementations with helper methods
- [x] Removed dead code from `_remove_by_sources()`
- [x] Cleaned up ISSUES.md, PLAN.md, README.md documentation
