# RecRAG - RAG Pipeline System

A modular Retrieval-Augmented Generation system with separate ingestion and retrieval pipelines. Process documents, generate embeddings, store them in FAISS, and query them using LLMs.

## Features

- **Dual Container Architecture**: Separate ingestion and retrieval containers
- **Auto-Ingestion**: Files uploaded via UI are automatically processed
- **Auto Model Pull**: Ollama models are automatically pulled on container startup
- **Multiple Providers**: Support for OpenAI and Ollama (local LLMs)
- **Vector Storage**: FAISS with file-based persistence
- **Flexible Configuration**: Environment variables with sensible defaults
- **Containerized**: Docker support for consistent deployment
- **Web UI**: Streamlit interface for document upload and querying

## Installation

### Prerequisites

- **Docker & Docker Compose** (recommended)
- **Python 3.12+** (for local development)
- **uv** package manager (for local development)
- **Memory**: At least 4GB RAM for Docker (6GB+ recommended for larger LLMs)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/RecRAG.git
   cd RecRAG
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Build and start**
   ```bash
   docker-compose -f docker/docker-compose.yml build
   docker-compose -f docker/docker-compose.yml up -d
   ```

4. **Access the UI**: http://localhost:8501

## Usage

### Docker with Ollama (Recommended)

```bash
# Create .env file
cat > .env << EOF
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BASE_URL=http://ollama:11434
LLM_PROVIDER=ollama
LLM_MODEL=granite3.1-moe:1b
LLM_BASE_URL=http://ollama:11434
EOF

# Build and start all services
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d

# Open http://localhost:8501
```

Models are automatically pulled by the `ollama-init` container. First startup may take a few minutes.

### Docker with OpenAI

```bash
# Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EOF

# Build and start (no Ollama needed)
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up -d ingestion retrieval

# Open http://localhost:8501
```

### Local Development

```bash
# Install dependencies
uv sync

# Terminal 1: Start Ollama (if using local models)
ollama serve

# Terminal 2: Start file watcher
uv run python backend/watch.py

# Terminal 3: Start Streamlit UI
uv run streamlit run backend/app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Ollama Container                           │
│  - Ollama server on port 11434                              │
│  - Models stored in Docker volume                           │
└─────────────────────────────────────────────────────────────┘
                              │
                    Healthcheck (healthy)
                              │
┌─────────────────────────────────────────────────────────────┐
│              Ollama-Init Container (one-time)               │
│  - Waits for Ollama to be healthy                           │
│  - Pulls embedding model (nomic-embed-text)                 │
│  - Pulls LLM model (granite3.1-moe:1b)                      │
│  - Exits after models are ready                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    Models ready
                              │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│  Ingestion Container    │         │  Retrieval Container    │
│  - watch.py daemon      │         │  - Streamlit UI         │
│  - Watches data/pdfs/   │◄───────►│  - Query interface      │
│  - Generates embeddings │ Shared  │  - FAISS search + LLM   │
│  - Updates status file  │ Volumes │                         │
└─────────────────────────┘         └─────────────────────────┘
```

### Container Paths

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | PDF uploads |
| `./storage` | `/app/storage` | FAISS index + status |
| `./config.toml` | `/app/config.toml` | Configuration |

### Network Communication

| Service | Address |
|---------|---------|
| Ollama | `http://ollama:11434` |
| Ingestion → Ollama | `http://ollama:11434` |
| Retrieval → Ollama | `http://ollama:11434` |

## Docker Commands

```bash
# Build all containers
docker-compose -f docker/docker-compose.yml build

# Start all services (recommended)
docker-compose -f docker/docker-compose.yml up -d

# Start specific services
docker-compose -f docker/docker-compose.yml up -d ollama          # Ollama only
docker-compose -f docker/docker-compose.yml up -d ingestion       # Ingestion only
docker-compose -f docker/docker-compose.yml up -d retrieval       # Retrieval only

# View logs
docker-compose -f docker/docker-compose.yml logs -f ollama
docker-compose -f docker/docker-compose.yml logs -f ingestion
docker-compose -f docker/docker-compose.yml logs -f retrieval

# Stop containers
docker-compose -f docker/docker-compose.yml down

# Remove all data (including models)
docker-compose -f docker/docker-compose.yml down -v
```

## Configuration

Configuration is managed via `config.toml` with environment variable substitution using `${VAR:-default}` syntax.

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | Embedding provider (`openai` or `ollama`) | `ollama` |
| `EMBEDDING_MODEL` | Embedding model name | `nomic-embed-text` |
| `EMBEDDING_BASE_URL` | Base URL for embedding service | `http://ollama:11434` |
| `LLM_PROVIDER` | LLM provider (`openai` or `ollama`) | `ollama` |
| `LLM_MODEL` | LLM model name | `granite3.1-moe:1b` |
| `LLM_BASE_URL` | Base URL for LLM service | `http://ollama:11434` |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) | `sk-...` |

### Configuration File

```toml
[embedding]
provider = "${EMBEDDING_PROVIDER:-openai}"
model = "${EMBEDDING_MODEL:-text-embedding-3-small}"
base_url = "${EMBEDDING_BASE_URL:-}"

[llm]
provider = "${LLM_PROVIDER:-openai}"
model = "${LLM_MODEL:-gpt-4o-mini}"
base_url = "${LLM_BASE_URL:-}"

[ingestion]
directory = "data/pdfs"
chunk_size = 1024
chunk_overlap = 50

[retrieval]
top_k = 4
context_template = """Context information:
{context}

Question: {question}

Answer:"""

[storage]
directory = "storage"
```

All paths are relative to the location of `config.toml`.

## Supported Models

### OpenAI
- **Embeddings**: `text-embedding-3-small`, `text-embedding-3-large`
- **LLMs**: `gpt-4o-mini`, `gpt-4o`, `gpt-4`

### Ollama (Local)

**Embeddings:**
- `nomic-embed-text` (274 MB) - Recommended
- `mxbai-embed-large` (669 MB)

**LLMs (by size):**
| Model | Size | Memory Required | Best For |
|-------|------|-----------------|----------|
| `tinyllama` | 0.6 GB | ~1.5 GB | Minimal resources |
| `granite3.1-moe:1b` | 1.4 GB | ~2.5 GB | Balanced |
| `llama3.2:1b` | 1.2 GB | ~2 GB | Fast responses |
| `llama3.2:3b` | 2 GB | ~4 GB | Better quality |
| `phi3:mini` | 2.2 GB | ~4 GB | Code tasks |
| `gemma2:2b` | 1.6 GB | ~3 GB | General purpose |

**Note:** Allocate at least 2x the model size in Docker memory.

## Troubleshooting

### "model requires more system memory"

The LLM model is too large for Docker's memory allocation.

**Solution:**
1. Open Docker Desktop → Settings → Resources
2. Increase Memory to 6GB or more
3. Restart Docker and containers

Or use a smaller model:
```env
LLM_MODEL=tinyllama
```

### "Unknown embedder provider: ${EMBEDDING_PROVIDER:-openai}"

Environment variables not being substituted.

**Solution:**
1. Ensure `.env` file exists in project root
2. Restart containers: `docker-compose -f docker/docker-compose.yml down && docker-compose -f docker/docker-compose.yml up -d`

### "404 Not Found for url: .../api/generate"

Model not found in Ollama.

**Solution:**
1. Check models: `docker exec recrag-ollama ollama list`
2. Pull missing model: `docker exec recrag-ollama ollama pull <model-name>`
3. Verify model name matches exactly (e.g., `granite3.1-moe:1b` not `granite3.1:moe:1b`)

### "Extra data: line 2 column 1 (char 105)"

Ollama streaming response issue.

**Solution:** Rebuild containers:
```bash
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml build --no-cache
docker-compose -f docker/docker-compose.yml up -d
```

### No documents found for querying

- Upload PDFs in the "Ingest Documents" tab
- Wait for status to show "complete"
- Check logs: `docker-compose -f docker/docker-compose.yml logs ingestion`

### Permission errors

```bash
chmod 777 storage/ data/
```

### Container won't start

```bash
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml logs
```

## Project Structure

```
RecRAG/
├── AGENTS.md                 # AI agent guidelines
├── CONTRIBUTING.md           # Contribution guidelines
├── README.md                 # This file
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Dependency lock file
├── config.toml               # Application configuration
├── .env.example              # Example environment variables
├── .env                      # Environment variables (not in git)
├── backend/                  # Python application code
│   ├── src/
│   │   ├── config.py         # Configuration loading & path resolution
│   │   ├── core.py           # Document processing & vector store
│   │   ├── pipelines.py      # Ingestion & retrieval pipelines
│   │   └── adapters/         # LLM & embedding providers
│   │       ├── base.py       # Abstract base classes
│   │       ├── embedding.py  # OpenAI + Ollama embedders
│   │       └── llm.py        # OpenAI + Ollama LLMs
│   ├── app.py                # Streamlit UI
│   ├── ingest.py             # CLI ingestion tool
│   └── watch.py              # File watcher daemon
├── docker/
│   ├── Dockerfile            # Multi-purpose Dockerfile
│   └── docker-compose.yml    # Container orchestration
├── docs/
│   └── ARCHITECTURE.md       # Detailed architecture
├── data/pdfs/                # PDF upload directory (not in git)
└── storage/                  # FAISS index & status files (not in git)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Detailed system design
- [Contributing](CONTRIBUTING.md) - Contribution guidelines
- [AGENTS.md](AGENTS.md) - Guidelines for AI coding agents
