# RecRAG - Outstanding Issues

Issues identified during code review, grouped by severity and category.

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

## Security Considerations

| Risk | Location | Status |
|------|----------|--------|
| File upload DoS | `app.py` | ✅ Fixed (size limits) |
| Path traversal | `app.py` | ✅ Fixed (filename sanitization) |
| Malformed PDFs | `app.py` | ✅ Fixed (magic byte check) |
| Prompt injection | `retrieval.py` | Open (complex to mitigate) |
| API key exposure | `.env` | Use secrets manager in production |

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
2. **Context Window Risk**: No token counting before LLM calls.

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
