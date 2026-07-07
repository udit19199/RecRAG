# Module 2: Architecture Overview

RecRAG is designed as three independent services with a clear 5-layer architecture. 

## The Services

| Service | Port | Technology | Job |
|---------|------|------------|-----|
| **Retrieval API** | `:8000` | FastAPI + uv | Answer questions (`POST /query`) |
| **Ingestion API** | `:8001` | FastAPI + uv | Upload and index PDFs (`POST /upload`) |
| **Frontend** | `:3000` | Next.js 16 | Chat UI + model comparison workbench |

There are **no hard dependencies between the services**. The Ingestion API writes to Milvus. The Retrieval API reads from
Milvus. The frontend calls both via HTTP. You can stop any service without crashing the others.

## The Five Layers (Backend)

Every backend feature in RecRAG follows this layered approach. 

```text
┌─────────────────────────────────────┐
│  app/          API (FastAPI)        │  ← HTTP routes, request validation
├─────────────────────────────────────┤
│  src/runtime/  Runtime              │  ← Lifecycle, hot-reload, concurrency
├─────────────────────────────────────┤
│  src/pipelines/ Pipelines           │  ← Business logic (retrieve + generate)
├─────────────────────────────────────┤
│  src/adapters/  Adapters            │  ← LLM & embedding provider wrappers
├─────────────────────────────────────┤
│  src/stores.py  Vector Store        │  ← Milvus (vector search)
└─────────────────────────────────────┘
```

Each layer **only depends on the layer directly below it**. An API route never talks to an adapter directly — it goes
through the runtime and pipeline.

Next up, let's look at the first two layers in [Module 3: APIs and Runtime](03_api_and_runtime.md).
