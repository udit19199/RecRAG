# RecRAG — Retrieval-Augmented Generation

Upload PDFs, index them, and ask questions via an LLM-backed RAG pipeline. Supports Ollama (local, zero-cost), OpenAI, Gemini, and NVIDIA NIM.

---

## Try it

**Prerequisites:** Python 3.12+, Node.js 20+, [uv](https://docs.astral.sh/uv/), pnpm (`npm install -g pnpm`), [Ollama](https://ollama.com/download).

```bash
git clone https://github.com/udit19199/RecRAG.git
cd RecRAG
make setup

# Pull models (swap for your preferred models)
ollama pull nomic-embed-text
ollama pull llama3.2

# Start everything — no API keys needed
make dev
```

| URL | What it is |
|-----|------------|
| http://localhost:3000 | Chat UI |
| http://localhost:8000 | Retrieval API docs |
| http://localhost:8001 | Ingestion API docs |
| http://localhost:8002 | Orchestrator API docs |

Upload a PDF, wait for it to index, then start asking questions.

> **Don't have Ollama?** Set a `GEMINI_API_KEY` in `.env` and change `provider` in `config.toml` to `"gemini"`.

---

## How it works

Four services, all started by `make dev`:

| Service | Port | Does what? |
|---------|------|------------|
| **Retrieval API** | `:8000` | Answers questions, serves config, lists models |
| **Ingestion API** | `:8001` | Accepts PDF uploads, reindexes, reports status |
| **Orchestrator API** | `:8002` | Recommendation intake, runs, and export |
| **Frontend** | `:3000` | Chat UI, model comparison, and recommendation flows |

Documents are parsed, chunked, embedded, and stored in **Milvus**. By default it uses **Milvus Lite** — no Docker needed.

---

## Providers

Edit `config.toml` to switch, or use the frontend model picker.

| Provider | API key needed? | Notes |
|----------|-----------------|-------|
| Ollama | None | Local, zero-cost. **Default.** |
| Gemini | `GEMINI_API_KEY` | Free tier available. |
| OpenAI | `OPENAI_API_KEY` | Cloud, fast |
| NVIDIA NIM | `NVIDIA_API_KEY` | Cloud API |

Keys go in `.env` (copy from `.env.example`).

---

## Commands

### Development

| Command | What it does |
|---------|-------------|
| `make dev` | Start all four services |
| `make retrieval` | Retrieval API only (`:8000`) |
| `make ingestion` | Ingestion API only (`:8001`) |
| `make orchestrator` | Orchestrator API only (`:8002`) |
| `make frontend` | Frontend only (`:3000`) |
| `make ingest` | One-shot ingestion of `data/pdfs/` |
| `make evaluate` | Run batch evaluation |

### Docker (optional)

| Command | What it does |
|---------|-------------|
| `make docker-up` | Full stack in Docker Compose |
| `make docker-down` | Stop the stack |
| `make infra-up` | Start Docker-based Milvus only |
| `make infra-down` | Stop Docker-based Milvus |
| `make smoke-test` | End-to-end smoke test against running services |

### Quality

| Command | What it does |
|---------|-------------|
| `make lint` | Ruff (backend) + Biome (frontend) |
| `make format` | Format both |
| `make test` | Run pytest |
| `make typecheck` | Run mypy |
| `make check` | Lint + typecheck + test |

---

## Configuration

Two files, one job:

- **`config.toml`** — everything non-sensitive (providers, models, chunk sizes, timeouts). Supports `${VAR:-default}` env-var substitution.
- **`.env`** — API keys and secrets only. Never committed.

Key `config.toml` sections:

| Section | Controls |
|---------|----------|
| `[embedding]` | Embedding provider, model, timeout |
| `[llm]` | LLM provider, model, timeout |
| `[vision]` | Vision extraction provider, model, timeout |
| `[ingestion]` | Chunk size, overlap, PDF directory |
| `[retrieval]` | top_k, max context tokens, prompt template |
| `[storage]` | Milvus mode (`lite` or `server`), collection prefix |

---

## Deployment

For a single-server deployment, use Docker Compose:

```bash
cp .env.example .env   # add your API keys
make docker-up
```

The Compose stack includes Milvus, both APIs, the orchestrator, and the frontend.

---

## Project structure

```
src/           Backend Python (pipelines, adapters, stores, orchestration)
app/           FastAPI route handlers (retrieval, ingestion, orchestrator)
frontend/      Next.js app
jobs/          One-shot scripts (ingest, evaluate)
tests/         pytest suite
```

See `AGENTS.md` for contributor guidelines and `ARCHITECTURE.md` for system design.
