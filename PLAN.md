# RecRAG Development Roadmap

Current state of the RecRAG system and planned improvements.

---

## Current Architecture

```
Entry Points          Pipelines              Components
─────────────         ─────────              ──────────
app.py (Streamlit) →  IngestionPipeline  →   PDFLoader
ingest.py (CLI)       RetrievalPipeline      TextSplitter (Sentence)
watch.py (daemon)                            FAISSVectorStore
                                             OpenAI/Ollama/NIM adapters
```

### Implemented Features
- Adapter pattern with registry for LLM/embedding providers
- Dependency injection in pipelines
- Incremental indexing with MD5 hash tracking
- Batch streaming ingestion (configurable batch size)
- File locking for concurrent FAISS access
- Connection pooling for HTTP adapters
- Environment variable substitution in config
- File upload validation (size, magic bytes, filename sanitization)
- Token counting with context truncation (tiktoken)
- FAISS index rebuild on source removal (no orphaned vectors)

### Test Coverage
- 28 tests in `tests/`
- Unit tests for core components (VectorStore, TextSplitter, DocumentLoader)
- Mock embedder/LLM fixtures for isolated testing

---

## Priority 1: High Priority

### 1. Configurable Timeouts

**Status**: Not Started | **Impact**: Production reliability

Timeouts hardcoded (30s embeddings, 120s LLM). No retry logic.

**Tasks**:
- Add timeout settings to `config.toml` under `[adapters]`
- Implement exponential backoff retry
- Add max retry configuration

**Files**: `backend/src/adapters/embedding.py`, `backend/src/adapters/llm.py`

---

### 2. Embedding Cache

**Status**: Not Started | **Impact**: API cost, performance

Re-embedding same queries/documents has no caching.

**Tasks**:
- Add in-memory LRU cache with configurable size
- Add cache hit/miss metrics for monitoring
- Consider persistent cache (SQLite) for production

**Files**: New `backend/src/cache.py` or in adapters

---

### 3. Atomic Status File Writes

**Status**: Not Started | **Impact**: Data corruption risk

Status file can be corrupted if process dies mid-write.

**Tasks**:
- Implement atomic writes (temp file + rename)
- Add status file validation on startup

**Files**: `backend/watch.py`

---

## Priority 2: Performance

### 4. Scalable FAISS Index

**Status**: Not Started | **Impact**: Search performance at scale

`IndexFlatL2` is O(n) brute-force. Slow at 100k+ vectors.

**Tasks**:
- Add `IndexIVFFlat` for approximate search
- Make index type configurable
- Add automatic index selection based on vector count

**Files**: `backend/src/stores/faiss.py`

---

### 5. Batch Metadata Writes

**Status**: Not Started | **Impact**: Ingestion performance

Every `add()` writes entire metadata JSON. For bulk ingestion, should batch.

**Tasks**:
- Add `flush()` method for explicit persistence
- Implement batched writes with configurable batch size
- Auto-flush on reaching batch limit

**Files**: `backend/src/stores/faiss.py`

---

### 6. Streamlit Non-Blocking Updates

**Status**: Not Started | **Impact**: UI responsiveness

Current polling blocks UI thread for up to 2 minutes.

**Tasks**:
- Replace polling with `st.rerun()` pattern
- Use `st.session_state` for status tracking
- Add manual refresh button

**Files**: `backend/app.py`

---

## Future Considerations

| Feature | Priority | Notes |
|---------|----------|-------|
| Hybrid search | Medium | Combine dense + sparse retrieval |
| Async execution | Medium | Non-blocking operations |
| Document deletion UI | Medium | Remove documents from index |
| Reranking | Medium | Improve retrieval quality |
| Evaluation framework | High | Compare pipeline performance |
| Metrics/monitoring | Medium | Prometheus/OpenTelemetry |
| Circuit breaker | Medium | Graceful degradation |

---

## Completed

- [x] Adapter pattern with registry
- [x] Dependency injection in pipelines
- [x] Incremental indexing
- [x] Batch streaming ingestion
- [x] File locking for concurrent access
- [x] Connection pooling
- [x] Environment variable substitution
- [x] Basic test coverage (28 tests)
- [x] FAISS index rebuild on source removal
- [x] BaseVectorStore import in retrieval pipeline
- [x] File upload validation (size, magic bytes, sanitization)
- [x] Token counting with context truncation
