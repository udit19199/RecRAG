# Architecture Deepening Candidates

Tracking file for architectural improvement opportunities identified on 2026-05-19.

---

## Candidate 1: Vector Store — Tight Milvus Coupling Without an Interface

**Cluster**: `stores.py` (`VectorStore`), `tests/mocks/memory_vector_store.py` (`InMemoryVectorStore`), `pipelines/ingestion.py`, `pipelines/retrieval.py`, `app/retrieval/main.py`, `app/ingestion/main.py`

**Status**: 🔵 In progress — problem framed, three interface designs proposed (see below)

**Why**: Production `VectorStore` wraps `MilvusClient` directly. Tests use a manually-duplicated `InMemoryVectorStore` with no shared protocol. Pipeline code can't accept either type polymorphically. Milvus-specific schema, filter syntax, and side-effectful construction leak into all callers.

**Dependency category**: Local-substitutable

**Preferred approach**: Factory function + Protocol (Design C hybrid). A `VectorStore` Protocol defines the read/write contract. A `create_vector_store` factory (in `pipelines/base.py`) centralizes construction complexity. Milvus implementation becomes a private detail behind the Protocol. Test implementation is in-memory alongside production code.

---

## Candidate 2: Pipeline Configuration — Distributed Creation Logic

**Cluster**: `pipelines/base.py` (adapter factories), `pipelines/ingestion.py:from_config()`, `pipelines/retrieval.py:from_config()`, `runtime/ingestion.py:_build_pipeline()`, `runtime/retrieval.py:_build_pipeline()`, `app/retrieval/main.py` (stateless query path)

**Status**: ⏳ Pending

**Why**: Pipeline creation logic is spread across 5+ locations, each independently interpreting config keys. The `config: dict[str, Any]` is a bag-of-parameters pattern. No single source of truth for what config a pipeline needs.

**Dependency category**: In-process

---

## Candidate 3: Document Loader Selection — Dense Conditional in Pipeline Construction

**Cluster**: `pipelines/ingestion.py:IngestionPipeline.from_config()`, `loaders.py`, `adapters/vision.py`

**Status**: ⏳ Pending

**Why**: Loader selection is a multi-way decision tree (ExtractionMode × DEV_MODE × try/except fallback) with different constructor signatures per loader. Untestable without instantiating the full pipeline.

**Dependency category**: In-process (plus Remote but owned for vision API)

---

## Candidate 4: Runtime Lifecycle — Duplicated Coordination Pattern

**Cluster**: `runtime/base.py` (protocol), `runtime/ingestion.py:IngestionRuntime`, `runtime/retrieval.py:RetrievalRuntime`

**Status**: ⏳ Pending

**Why**: Both runtimes independently implement the same lifecycle machinery (lock management, in-flight tracking, drain events, shutdown). `_drain_and_close()` is verbatim copy-pasted. The `PipelineRuntime` protocol provides no shared implementation.

**Dependency category**: In-process

---

## Performance: Dependency Bloat Cleanup (2026-05-19)

**Changes made**:
1. **Replaced ragas with direct GPT calls** in `src/evaluation/ragas_eval.py` — dropped `ragas`, `datasets`, `langchain-openai`, all `langchain*`, and **`torch` (376 MB)**
2. **Removed `google-generativeai`** from `pyproject.toml` — dropped `google-api-python-client` (99 MB) — the package was never actually imported; our Gemini adapter uses raw HTTP
3. **Cleaned up `langchain-openai`** from deps — was unused

**Results**: 1.3 GB → **632 MB** venv (51% reduction), 184 → 132 packages
