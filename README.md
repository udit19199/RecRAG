# RecRAG — Retrieval-Augmented Generation

A lightweight, multi-provider RAG pipeline: upload PDFs, index the corpus, and query via LLM-backed retrieval.

**Ollama is a first-class citizen** — local, air-gapped, zero-cost operation is the default. OpenAI, Gemini, and NVIDIA NIM are also supported.

---

## Architecture

Two FastAPI services + a Next.js frontend:

| Service | Port | Purpose |
|---------|------|---------|
| **Retrieval API** | `:8000` | Query endpoint, config, model discovery, evaluation |
| **Ingestion API** | `:8001` | Upload PDFs, reindex, check status |
| **Frontend** | `:3000` | Chat UI + model comparison workbench |

Documents are parsed, chunked, embedded, and stored in **Milvus** (Lite for local dev or server/cloud).

---

## Prerequisites

- **Python 3.12+**
- **Node.js 20+** and **pnpm** (`npm install -g pnpm`)
- **uv** (package manager) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker & Docker Compose** (optional — only for Milvus server mode or full Docker stack)
- **Ollama** (recommended for local dev) — [ollama.com/download](https://ollama.com/download)

---

## Quick Start (Local Dev)

### 1. Clone and set up

```bash
git clone https://github.com/udit19199/RecRAG.git
cd RecRAG
make setup
```

This installs Python + Node dependencies and creates a `.env` file from `.env.example`.

### 2. Pull models (Ollama)

```bash
ollama pull nomic-embed-text   # embedding
ollama pull llama3.2            # LLM (or any other model)
```

### 3. Start everything

```bash
make dev
```

| URL | Service |
|-----|---------|
| http://localhost:3000 | Frontend |
| http://localhost:8000 | Retrieval API docs |
| http://localhost:8001 | Ingestion API docs |

Upload a PDF, wait for indexing, then start asking questions.

### Default configuration

The `config.toml` ships with Ollama defaults — no API keys needed for local use:

```toml
[embedding]
provider = "ollama"
model = "nomic-embed-text"

[llm]
provider = "ollama"
model = "llama3.2"

[vision]
provider = "ollama"
model = "llama3.2-vision"
```

---

## Other Providers

Switch by editing `config.toml` or using the frontend model picker:

| Provider | Env Var Required | Notes |
|----------|-----------------|-------|
| Ollama | None | Local, zero-cost |
| OpenAI | `OPENAI_API_KEY` | Cloud, fast |
| Gemini | `GEMINI_API_KEY` | Free tier available |
| NVIDIA NIM | `NVIDIA_API_KEY` | Cloud API |

Set the required keys in `.env`:

```env
# At least one provider
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
NVIDIA_API_KEY=nvapi-...

# Optional: override Ollama host (default http://localhost:11434)
OLLAMA_HOST=http://192.168.1.100:11434
```

---

## Commands Reference

### Development

| Command | What it does |
|---------|-------------|
| `make dev` | Start retrieval (`:8000`) + ingestion (`:8001`) + frontend (`:3000`) |
| `make retrieval` | Start only the retrieval API |
| `make ingestion` | Start only the ingestion API |
| `make frontend` | Start only the Next.js dev server |
| `make ingest` | One-shot ingestion of all PDFs in `data/pdfs/` |
| `make evaluate` | Run batch evaluation on dataset |

### Infra

| Command | What it does |
|---------|-------------|
| `make infra-up` | Start local Milvus stack (Docker) |
| `make infra-down` | Stop local Milvus stack |
| `make docker-up` | Full stack in Docker |
| `make docker-down` | Stop full Docker stack |

### Quality

| Command | What it does |
|---------|-------------|
| `make lint` | Ruff (backend) + Biome (frontend) |
| `make format` | Format both backend and frontend |
| `make test` | Run pytest |
| `make typecheck` | Run mypy |
| `make check` | Lint + typecheck + test |

### Deploy

| Command | What it does |
|---------|-------------|
| `make deploy` | Build Docker images, import into k3s, deploy |
| `make deploy-dev` | Same as deploy with dev overlay (smaller resources) |
| `make smoke-test` | Run smoke tests against running services |

---

## Deployment

### Kubernetes (Recommended)

Deploy on any CNCF cluster (k3s, minikube, GKE, EKS, AKS) using Kustomize:

```bash
cp .env.example .env  # Edit with your API keys
make deploy           # One command: builds images, installs k3s if needed, deploys
```

Advanced variants:

```bash
make deploy-dev                          # Dev overlay (smaller resources)
DOMAIN=203.0.113.42.nip.io make deploy   # Custom domain / nip.io
```

### Docker Compose

```bash
cp .env.example .env  # Edit with your API keys
make docker-up
```

---

## Configuration

All non-sensitive config lives in `config.toml` (supports `${VAR:-default}` env var substitution).
Sensitive values (API keys) go in `.env` only — never commit them.

### Key `config.toml` sections

| Section | Purpose |
|---------|---------|
| `[embedding]` | Embedding provider, model, timeout, base_url |
| `[llm]` | LLM provider, model, timeout, base_url |
| `[vision]` | Vision extraction provider, model, timeout, base_url |
| `[ingestion]` | Chunk size, overlap, PDF directory |
| `[retrieval]` | top_k, max context tokens, prompt template |
| `[storage]` | Milvus deployment mode (lite/server), collection prefix |
| `[storage.server]` | Milvus server host, port, credentials |

Each Ollama-using section can set `base_url` to point at a different Ollama host:

```toml
[llm]
provider = "ollama"
model = "llama3.2"
base_url = "http://192.168.1.100:11434"
```

When `OLLAMA_HOST` env var is set, it overrides all Ollama base URLs.

---

## Documentation

| File | What it covers |
|------|---------------|
| `AGENTS.md` | Coding guidelines, style guide, detailed commands (for human + AI contributors) |
| `ARCHITECTURE.md` | System architecture, data flow diagrams, component inventory |

---

## Development Notes

- Backend: Python 3.12, FastAPI, uv, pytest, Ruff, mypy, Milvus
- Frontend: Next.js 16, React 19, TypeScript, Tailwind, Biome
- Backend code in `src/`, `app/`, `jobs/`
- Frontend code in `frontend/`
- Tests in `tests/`
