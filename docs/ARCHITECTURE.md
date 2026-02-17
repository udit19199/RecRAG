# RecRAG Architecture

## System Overview

RecRAG is a modular Retrieval-Augmented Generation (RAG) system with a dual-container architecture:

```
┌─────────────────────────────────────────────────────────────┐
│           Ingestion Container (recrag-ingestion)            │
│  - Runs watch.py (file watcher daemon)                      │
│  - Watches data/pdfs/ for new PDF files                     │
│  - Generates embeddings using configured provider           │
│  - Stores vectors in FAISS index                            │
│  - Writes status to storage/ingestion_status.json           │
└─────────────────────────────────────────────────────────────┘
                              │
                    Shared Volumes (data/, storage/)
                              │
┌────────────────────────────────────────────────────────────┐
│           Retrieval Container (recrag-retrieval)           │
│  - Runs app.py (Streamlit UI)                              │
│  - File upload and querying interface                       │
│  - Polls ingestion_status.json for progress                │
│  - Searches FAISS index for relevant context               │
│  - Uses LLM to generate answers                            │
└────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
RecRAG/
├── AGENTS.md                 # AI agent guidelines
├── README.md                 # Quick start guide
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Dependency lock file
├── config.toml               # Application configuration
├── .env                      # Environment variables
├── backend/                  # Python application code
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration loading & path resolution
│   │   ├── core.py           # Document loading, splitting, vector store
│   │   ├── pipelines.py      # Ingestion + retrieval pipelines
│   │   └── adapters/         # LLM & embedding providers
│   │       ├── __init__.py   # Factory functions
│   │       ├── base.py       # Abstract base classes
│   │       ├── embedding.py  # OpenAI + Ollama embedders
│   │       └── llm.py        # OpenAI + Ollama LLMs
│   ├── app.py                # Streamlit UI entry point
│   ├── ingest.py             # CLI ingestion tool
│   └── watch.py              # File watcher daemon
├── docker/
│   ├── Dockerfile            # Multi-purpose Dockerfile
│   └── docker-compose.yml    # Container orchestration
├── docs/
│   └── ARCHITECTURE.md       # This file
├── data/pdfs/                # PDF upload directory (shared)
└── storage/                  # FAISS index + status files (shared)
```

## Container Paths

Both containers use `/app/` as the working directory:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | PDF uploads |
| `./storage` | `/app/storage` | FAISS index + status |
| `./config.toml` | `/app/config.toml` | Configuration |

## Module Breakdown

### Adapters (`backend/src/adapters/`)

The adapter pattern provides abstraction over different LLM and embedding providers.

#### base.py
- `BaseEmbedder`: Abstract class for embedding providers
- `BaseLLM`: Abstract class for LLM providers

#### embedding.py
- `OpenAIEmbedder`: OpenAI embeddings API
- `OllamaEmbedder`: Ollama local embeddings

#### llm.py
- `OpenAILLM`: OpenAI Chat API
- `OllamaLLM`: Ollama local LLM

**Adding a new provider:**
1. Implement `BaseEmbedder` or `BaseLLM` in appropriate file
2. Add to factory function in `__init__.py`

### Config (`backend/src/config.py`)

Configuration management with environment variable substitution and path resolution.

- `load_config()`: Load TOML config with `${VAR:-default}` syntax
- `resolve_path()`: Resolve paths relative to config file location
- `get_config_value()`: Get nested config using dot notation

### Core (`backend/src/core.py`)

Core document processing components.

#### DocumentLoader
Loads PDF files using llama-index:
- `load()`: Load all documents from directory
- `load_file()`: Load single file

#### TextSplitter
Chunks documents using sentence-level splitting:
- `split_documents()`: Split multiple documents
- `split_text()`: Split single text

#### VectorStore
FAISS-based vector storage with persistence:
- `add()`: Add embeddings to index
- `search()`: Find similar vectors
- `save()`: Persist index to disk

### Pipelines (`backend/src/pipelines.py`)

High-level workflows combining multiple components.

#### IngestionPipeline
1. Load documents from `data/pdfs/`
2. Split into chunks
3. Generate embeddings
4. Store in FAISS

#### RetrievalPipeline
1. Embed query
2. Search FAISS for relevant chunks
3. Generate response using LLM

## Entry Points

### app.py (Streamlit UI)
- Web interface for testing/demonstration
- File upload + query in one UI
- Polls status file for ingestion progress

### ingest.py (CLI)
- One-time ingestion script
- Use for batch processing

### watch.py (File Watcher)
- Daemon that watches for new files
- Auto-triggers ingestion
- Writes status to JSON file

## Configuration System

The system uses TOML configuration with environment variable substitution:

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

[storage]
directory = "storage"
```

**Supported environment variables:**
- `EMBEDDING_PROVIDER`: "openai" or "ollama"
- `EMBEDDING_MODEL`: Embedding model name
- `EMBEDDING_BASE_URL`: Custom endpoint
- `LLM_PROVIDER`: "openai" or "ollama"
- `LLM_MODEL`: LLM model name
- `LLM_BASE_URL`: Custom endpoint
- `OPENAI_API_KEY`: OpenAI API key

**Path Resolution:**
All paths in config.toml are relative to the config file's parent directory. The `resolve_path()` function handles this:

```python
from config import resolve_path

# Resolves "data/pdfs" relative to config file location
pdf_dir = resolve_path("data/pdfs", config_path)
```

## File Watcher Protocol

The ingestion container and UI communicate via a status file:

**Status file:** `storage/ingestion_status.json`

```json
{
  "status": "processing|complete|error|idle",
  "started_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:35:00",
  "files_processed": 3,
  "error_message": null
}
```

**Workflow:**
1. UI uploads files to shared `data/pdfs/` volume
2. Watcher detects new files
3. Watcher sets status to "processing"
4. Ingestion runs, updates status to "complete" or "error"
5. UI polls and shows progress

## Docker Architecture

### Container Separation

**Ingestion Container (recrag-ingestion):**
- Runs `watch.py` (file watcher daemon)
- Mounts `data/` and `storage/`
- Needs only embedding provider config

**Retrieval Container (recrag-retrieval):**
- Runs Streamlit UI
- Mounts same volumes
- Needs both embedding + LLM config

**Ollama Container (recrag-ollama):**
- Optional, for local models
- Exposes port 11434

### Docker Commands

```bash
# Build all containers
docker-compose -f docker/docker-compose.yml build

# Start ingestion container only
docker-compose -f docker/docker-compose.yml up ingestion

# Start retrieval container only
docker-compose -f docker/docker-compose.yml up retrieval

# Start both containers
docker-compose -f docker/docker-compose.yml up

# Start all services including Ollama
docker-compose -f docker/docker-compose.yml up ollama ingestion retrieval

# Run in background
docker-compose -f docker/docker-compose.yml up -d
```

## Development Workflow

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Start Ollama (optional):**
   ```bash
   docker-compose -f docker/docker-compose.yml up -d ollama
   docker exec -it recrag-ollama ollama pull nomic-embed-text
   docker exec -it recrag-ollama ollama pull granite3.3:latest
   ```

3. **Run containers:**
   ```bash
   # Both containers
   docker-compose -f docker/docker-compose.yml up

   # Or locally
   uv run python backend/watch.py        # Terminal 1
   uv run streamlit run backend/app.py   # Terminal 2
   ```

4. **Access UI:** http://localhost:8501
