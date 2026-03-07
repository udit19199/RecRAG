# RecRAG — Retrieval-Augmented Generation

Lightweight RAG pipeline: upload a PDF batch, index the full corpus, and query via LLM-backed RAG.

Quick links
- Docs: [docs/](docs/)
- Troubleshooting: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

Quick start (Docker)
```bash
cp .env.example .env
docker-compose build
docker-compose up -d
```

Local dev
```bash
# Backend deps
uv sync
uv pip install -e .

# Start APIs (use --env-file .env to set PYTHONPATH)
uv run --env-file .env uvicorn api.retrieval.main:app --port 8000 &
uv run --env-file .env uvicorn api.ingestion.main:app --port 8001 &

# Frontend
cd frontend && pnpm dev
```

Service URLs
- Frontend: http://localhost:3000
- Retrieval API: http://localhost:8000
- Ingestion API: http://localhost:8001

Testing
```bash
uv run pytest
```

Notes
- Configuration sits in `config.toml` (supports ${VAR:-default} substitution).
- For full developer guidance see files under `docs/` and `AGENTS.md`.

Contributing
- See `CONTRIBUTING.md` for commit conventions and development workflow.
