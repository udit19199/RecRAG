# RecRAG — Retrieval-Augmented Generation

Upload PDFs, index them, and ask questions via an LLM-backed RAG pipeline. Supports Ollama (local, zero-cost), OpenAI, Gemini, and NVIDIA NIM.

---

## Try it

**Prerequisites** (you probably have these): Python 3.12+, Node.js 20+, [uv](https://docs.astral.sh/uv/), pnpm (`npm install -g pnpm`), [Ollama](https://ollama.com/download).

```bash
# 1. Clone and install dependencies
git clone https://github.com/udit19199/RecRAG.git
cd RecRAG
make setup

# 2. Pull models (swap for your preferred models)
ollama pull nomic-embed-text
ollama pull llama3.2

# 3. Start everything — no API keys needed
make dev
```

| URL | What it is |
|-----|------------|
| http://localhost:3000 | Chat UI |
| http://localhost:8000 | Retrieval API docs |
| http://localhost:8001 | Ingestion API docs |

Upload a PDF, wait for it to index, then start asking questions.

> **Don't have Ollama?** Set a `GEMINI_API_KEY` in `.env` and change `provider` in `config.toml` to `"gemini"`.

---

## How it works

Three services, all started by `make dev`:

| Service | Port | Does what? |
|---------|------|------------|
| **Retrieval API** | `:8000` | Answers questions, serves config, lists models |
| **Ingestion API** | `:8001` | Accepts PDF uploads, reindexes, reports status |
| **Frontend** | `:3000` | Chat UI + model comparison workbench |

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

Keys go in `.env` (copy from `.env.example`). [Read more about configuration](#configuration).

---

## Commands

### Development

| Command | What it does |
|---------|-------------|
| `make dev` | Start all three services |
| `make retrieval` | Retrieval API only (`:8000`) |
| `make ingestion` | Ingestion API only (`:8001`) |
| `make frontend` | Frontend only (`:3000`) |
| `make ingest` | One-shot ingestion of `data/pdfs/` |
| `make evaluate` | Run batch evaluation |

### Infra (optional — only if you need Docker-based Milvus)

| Command | What it does |
|---------|-------------|
| `make infra-up` | Start local Milvus (Docker) |
| `make infra-down` | Stop it |
| `make docker-up` | Full stack in Docker |
| `make docker-down` | Stop full stack |

### Quality

| Command | What it does |
|---------|-------------|
| `make lint` | Ruff (backend) + Biome (frontend) |
| `make format` | Format both |
| `make test` | Run pytest |
| `make typecheck` | Run mypy |
| `make check` | Lint + typecheck + test |

### Deploy

| Command | What it does |
|---------|-------------|
| `make docker-up` | Run the app stack locally with Docker Compose |
| `make deploy` | Build → k3s → deploy (legacy Kubernetes path) |
| `make deploy-dev` | Same, dev overlay (smaller resources) |

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

To switch from Ollama to Gemini:

```toml
[embedding]
provider = "gemini"
model = "text-embedding-004"

[llm]
provider = "gemini"
model = "gemini-2.0-flash"
```

To point Ollama at a different host:

```toml
[llm]
base_url = "http://192.168.1.100:11434"
```

Set `OLLAMA_HOST` in `.env` to override all Ollama base_urls at once.

---

## Deployment

### EC2 with CI/CD (recommended for small teams)

For a production setup with a single EC2 host and automated deploys:

- use `docker-compose.ec2.yml`
- publish images to ECR from GitHub Actions
- deploy to EC2 over SSH with `scripts/ec2/deploy.sh`

See [docs/AWS_EC2_CICD.md](docs/AWS_EC2_CICD.md) for the full setup.

### Kubernetes

```bash
cp .env.example .env   # add your API keys
make deploy            # builds images, installs k3s if needed, deploys
```

Variants: `make deploy-dev`, `DOMAIN=203.0.113.42.nip.io make deploy`

### Docker Compose

```bash
cp .env.example .env
make docker-up
```

---

## Project structure

```
src/           ← Backend Python (pipelines, adapters, stores, config)
app/           ← FastAPI route handlers (retrieval + ingestion)
frontend/      ← Next.js 16 app
jobs/          ← One-shot scripts (ingest, evaluate)
tests/         ← pytest suite
k8s/           ← Kubernetes manifests (Kustomize)
```

See `AGENTS.md` for contributor guidelines and `ARCHITECTURE.md` for system design.
