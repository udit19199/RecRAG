# RecRAG - Outstanding Issues

This document tracks outstanding code quality, performance, and architecture issues identified during code review.

---

## 🔴 Critical

### 1. No Tests Exist

Despite `pyproject.toml` and `AGENTS.md` referencing `pytest`, there are zero test files. This is a critical gap for any production system.

**Recommended Actions:**
- Add unit tests for `core.py`, `pipelines.py`, `adapters/`
- Add integration tests for end-to-end validation
- Set up CI/CD pipeline with test coverage requirements

---

### 2. Data Loss in Parallel Embedding

**File:** `embedding.py:109`

```python
return [r for r in results if r is not None]
```

If any embedding fails in `OllamaEmbedder.embed_batch()`, it's silently dropped. This causes:
- Index misalignment between embeddings and chunks
- Silent data corruption
- Vector store will have fewer vectors than expected

**Recommended Actions:**
- Raise exception if any embedding fails with index information
- Implement partial failure handling with rollback
- Add logging for failed embeddings

---

### 3. Dead Code in `_remove_by_sources()` Index Rebuild

**File:** `core.py:126-128`

```python
if "embedding" in m:
    remaining_embeddings.append(m["embedding"])
```

The `_remove_by_sources()` method attempts to rebuild the FAISS index by looking for `"embedding"` keys in metadata. However, metadata only stores `text`, `index`, and custom fields like `source` — **embeddings are never stored in metadata**. This code path never executes, meaning:
- Index rebuilding after source removal doesn't work
- The rebuilt index will be empty even if metadata has entries

**Recommended Actions:**
- Store embeddings in metadata when adding documents
- OR re-embed remaining texts (slower but correct)
- OR use a different deletion strategy (mark as deleted, don't rebuild)

---

## 🟠 High Priority

### 4. Placeholder Project Metadata

**File:** `pyproject.toml:1-4`

```toml
name = "test-app"
version = "0.1.0"
description = "Add your description here"
```

Project still has placeholder values from initial setup.

**Recommended Actions:**
- Update name to "recrag"
- Write proper description
- Add authors, license, and repository fields

---

### 5. No Dev Dependencies

**File:** `pyproject.toml`

The following tools are documented in `AGENTS.md` but not installable:
- `pytest` - testing
- `ruff` - linting/formatting
- `mypy` - type checking

**Recommended Actions:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
]
```

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

### 14. No Connection Pooling

**Files:** `embedding.py`, `llm.py`

Creating new `requests` calls without connection pooling wastes resources.

**Recommended Actions:**
- Use `requests.Session` with connection pooling for Ollama
- OpenAI client reuse is already implemented
- Consider aiohttp for async requests

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
