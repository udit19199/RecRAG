# RecRAG Development Roadmap

Future work for the RecRAG system.

---

## Verified Open Issues

### 1. Ingestion upload blocks the event loop

**Status**: Open | **Impact**: Ingestion throughput

`app/ingestion/main.py` still writes uploaded PDFs with synchronous file I/O inside an `async` route.

**Tasks**:
- Replace direct file writes with async-friendly or threaded writes
- Keep batch replacement behavior intact
- Add a regression test for concurrent upload/status handling

### 2. Adapter timeouts are hardcoded

**Status**: Open | **Impact**: Production reliability

Embedding and LLM adapters still hardcode request timeouts.

**Tasks**:
- Move timeout settings to `config.toml`
- Preserve current defaults
- Add config parsing tests

### 3. Retry policy can replay POSTs

**Status**: Open | **Impact**: Duplicate ingestion requests

`src/adapters/utils.py` still uses `HTTPAdapter(max_retries=...)` directly.

**Tasks**:
- Switch to `urllib3.util.Retry`
- Restrict retries to idempotent methods
- Add tests for adapter construction

### 4. Token counting fallback is inaccurate

**Status**: Open | **Impact**: Context truncation errors

`src/pipelines/retrieval.py` still falls back to `cl100k_base` for unknown models.

**Tasks**:
- Make the fallback explicit in config or model mapping
- Document expected accuracy limits
- Add a test for non-OpenAI models

### 5. NIM embedder probes the API in `__init__`

**Status**: Open | **Impact**: Startup latency and testability

`src/adapters/nim.py` still makes a live call to detect embedding dimensions.

**Tasks**:
- Use a lazy dimension lookup or config override
- Preserve compatibility with existing models
- Add a unit test for constructor behavior

### 6. API keys can remain in `self.kwargs`

**Status**: Open | **Impact**: Secret exposure risk

Base adapter constructors still store kwargs before OpenAI keys are removed.

**Tasks**:
- Pop secrets before calling `super().__init__()`
- Avoid storing credentials in `self.kwargs`
- Add a regression test for adapter state

### 7. No document deletion API

**Status**: Open | **Impact**: Lifecycle management gap

The system still has no endpoint for deleting a document from the state record and the vector index.

**Tasks**:
- Define deletion semantics for file and vector-store cleanup
- Add a DELETE route and tests
- Update the frontend/API client if needed

### 8. API authentication is missing

**Status**: Open | **Impact**: Unauthenticated access

No bearer-token or similar auth guard exists on the APIs.

**Tasks**:
- Add a lightweight auth mechanism
- Apply it consistently across both services
- Document env vars and local-dev behavior

## Security Considerations

| Risk | Status |
|------|--------|
| Unauthenticated Access | Open |

## Issue Notes

- Path traversal is fixed via filename sanitization in the upload route.
- File type validation is fixed via the PDF extension check.
- CORS is fixed through configured frontend origins in `config.toml`.
- Status writes are now atomic via temp-file replacement.
- CORS is now restricted through `config.toml` frontend origins.
- MD5-based change detection is not present in the current codebase.

## Execution Plan

1. Fix ingestion and status persistence first.
2. Fix adapter retry, timeout, and secret handling.
3. Address auth.
4. Fix retrieval/NIM edge cases.
5. Add deletion support if state contracts are clear.
6. Run targeted tests and update the roadmap notes.

## Future Considerations

| Feature | Priority | Notes |
|---------|----------|-------|
| Hybrid search | Medium | Combine dense + sparse retrieval |
| Reranking | Medium | Improve retrieval quality |
| Evaluation framework | High | Compare pipeline performance |
| Metrics/monitoring | Medium | Prometheus/OpenTelemetry |
| Circuit breaker | Medium | Graceful degradation |
