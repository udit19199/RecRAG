# RecRAG — Retrieval-Augmented Generation

Lightweight RAG pipeline: upload a PDF batch, index the full corpus, and query via LLM-backed RAG.

## Quick Start (Docker)

```bash
cp .env.example .env  # Update with your API keys and Milvus credentials
make docker-up
```

The default and recommended setup is managed/serverless Milvus configured via
`config.toml` and `.env`.

## Local Development

```bash
make setup      # One-time setup (deps + .env)
make dev        # Start all services (Next.js + APIs)
```

Optional local Milvus for development:

```bash
make infra-up
```

- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **APIs**: [8000](http://localhost:8000) (Retrieval) / [8001](http://localhost:8001) (Ingestion)
- **Checks**: `make check` (Lint + Test)
- **Tools**: `make ingest` / `make evaluate`

---

## Configuration
Configuration is managed in `config.toml` (supports `${VAR:-default}` env var substitution).
