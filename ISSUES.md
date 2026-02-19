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

**File**: `backend/app.py:131-145`

```python
for _ in range(60):
    time.sleep(2)  # Blocks UI thread for 2 minutes!
```

**Impact**: UI frozen, poor user experience.

**Solution**: Use `st.rerun()` with `st.session_state` for non-blocking updates.

---

### 3. Non-Atomic Status File Writes

**File**: `backend/watch.py:71-72`

```python
with open(self.status_file, "w") as f:
    json.dump(status_data, f)
```

Process crash mid-write corrupts the file.

**Solution**: Write to temp file, then atomic rename.

---

## Medium Priority

### 4. No Embedding Cache

Same query re-embedded every time. No caching at any level.

**Impact**: Wasted API calls, slower responses.

**Solution**: Add LRU cache with configurable size (default 1000 entries).

---

### 5. Full JSON Write on Every Add

**File**: `backend/src/stores/faiss.py:102`

Every `add()` writes entire metadata JSON to disk. For bulk ingestion of 10k documents, that's 10k full file writes.

**Solution**: Batch writes with explicit `flush()` or auto-flush at batch limit.

---

### 6. IndexFlatL2 Doesn't Scale

**File**: `backend/src/stores/faiss.py:29`

`IndexFlatL2` is brute-force O(n) search. At 100k+ vectors, search becomes noticeably slow.

**Solution**: Use `IndexIVFFlat` for approximate search at scale.

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

## Test Coverage Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| Core (VectorStore, Splitter, Loader) | Tested | 11 tests |
| Adapters (OpenAI, Ollama) | Tested | 17 tests with mocks |
| Pipelines | Missing | No tests |
| Watch daemon | Missing | No tests |
| Config edge cases | Missing | No tests |
| Integration (E2E) | Missing | No tests |

---

## Performance Notes

1. **Large corpora**: Current implementation loads all documents into memory during full ingestion. Use `process_documents_streaming()` for lower memory usage.

2. **Concurrent requests**: No request queueing. High load could overwhelm the system.

3. **Index rebuilds**: ✅ Fixed - Source removal now rebuilds FAISS index to remove orphaned vectors.
