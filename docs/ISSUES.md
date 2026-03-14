# RecRAG - Outstanding Issues

Issues identified during code review, updated to reflect the current root-level project structure.

---

## Critical Priority

### C1. Blocking Filesystem I/O Inside `async` Upload Route

**File**: `api/ingestion/main.py:216` — `upload_pdfs()`

```python
for safe_filename, content in staged_files:
    file_path = PDF_DIR / safe_filename
    with open(file_path, "wb") as f:
        f.write(content)  # synchronous, blocks the event loop
```

`f.write(content)` is a synchronous call inside an `async def` route handler. This blocks the asyncio event loop, preventing the service from handling other concurrent requests (like `/status` or `/health`) while the disk write is in progress.

**Impact**: Large uploads stall the entire ingestion service for all users.

**Solution**: Use `anyio.to_thread.run_sync` or `aiofiles`:
```python
import aiofiles

async with aiofiles.open(file_path, "wb") as f:
    await f.write(content)
```

---

## High Priority

### 1. Hardcoded Timeouts in Adapters

**Files**: `src/adapters/embedding.py:78, 110`, `src/adapters/llm.py:101, 115`

- 30s for standard embeddings
- 120s for LLM/Batch embeddings

These are hardcoded in `.post()` calls and are not configurable via `config.toml`.

**Solution**: Move timeouts to `config.toml` under a `[network]` or `[adapters]` section.

---

### 2. Non-Atomic Status File Writes

**File**: `src/utils/status.py:46` — `write_status()`

```python
with open(get_status_file(storage_dir), "w") as f:
    json.dump(data, f, indent=2)
```

If the process crashes or the disk fills up during this write, `ingestion_status.json` will be left in a corrupted/partial state.

**Solution**: Write to a temporary file in the same directory and use `os.replace()` for an atomic rename.

---

### 3. HTTP Retry Strategy Retries Non-Idempotent POSTs

**File**: `src/adapters/utils.py:12` — `create_session_with_pooling()`

The `max_retries` integer passed to `HTTPAdapter` defaults to retrying all methods, including POST.

**Impact**: Transient errors during embedding or generation can result in duplicate requests. While LLM generation is mostly harmless (just extra cost/latency), retrying vector ingestion POSTs can lead to duplicate entries in Milvus.

**Solution**: Use a `urllib3.util.Retry` object with `allowed_methods=["GET"]`.

---

### 4. Wildcard CORS in Production

**Files**: `api/ingestion/main.py:137`, `api/retrieval/main.py:76`

```python
allow_origins=["*"]
```

**Impact**: Allows any domain to make browser-based requests to the APIs.

**Solution**: Restrict `allow_origins` to the specific frontend URL via environment variables.

---

## Medium Priority

### 1. `tiktoken` Accuracy for Non-OpenAI Models

**File**: `src/pipelines/retrieval.py:32` — `_count_tokens()`

The code uses `cl100k_base` (OpenAI) as a fallback for all models, including Llama-3 (Ollama) or NIM models. Tokenization differs significantly between these families.

**Impact**: Context window calculations in `RetrievalPipeline.generate` will be inaccurate by up to 30%, potentially leading to context overflow errors or aggressive over-truncation.

---

### 2. NIMEmbedder Live API Call in Constructor

**File**: `src/adapters/nim.py:44` — `NIMEmbedder.__init__()`

The constructor calls `self._client.get_query_embedding("test")` to detect dimensions.

**Impact**: Instantiating the class requires a valid API key and internet connection. This makes unit testing difficult and slows down application startup.

**Solution**: Use a lazy property for `dimension` or allow it to be passed as an optional argument in `config.toml`.

---

### 3. API Key persistence in `self.kwargs`

**Files**: `src/adapters/embedding.py:27`, `src/adapters/llm.py:23`

In OpenAI adapters, `api_key` is popped from `kwargs` *after* `super().__init__(model, **kwargs)` is called. The base class stores `self.kwargs = kwargs`.

**Impact**: Sensitive API keys may be stored in the object's `self.kwargs` dictionary, potentially leaking into logs if the object is inspected or serialized.

---

## Low Priority

### 1. No Document Deletion API

The system currently supports upload and re-indexing, but there is no endpoint to delete a specific document from the Milvus collection or the `data/pdfs` storage.

### 2. MD5 for Change Detection

**File**: `src/pipelines/ingestion.py` (referenced logic)

Uses MD5 for file hashing. While sufficient for simple change detection, SHA-256 is generally preferred for modern applications to avoid collision risks and comply with stricter security policies.

---

## Security Considerations

| Risk | Status |
|------|--------|
| Path Traversal | ✅ Fixed via `Path(upload.filename).name` in `api/ingestion/main.py` |
| File Type Validation | ✅ Fixed via `.endswith(".pdf")` check |
| API Key Exposure | ⚠️ Partial - Keys are in `.env`, but see Medium Issue #3 regarding `self.kwargs` |
| Unauthenticated Access | ❌ Open - API endpoints do not have Auth/Bearer token checks |