# RAG-Architect: Pipeline Recommendation System

RAG-Architect is a specialized system designed to recommend and implement optimal Retrieval-Augmented Generation (RAG) pipelines. It handles the end-to-end process of ingesting documents, evaluating different embedding/retrieval strategies, and providing recommendations for the best-performing pipeline configuration.

## Features

- **Document Ingestion:** Automatically reads PDFs and prepares them for vectorization.
- **Pipeline Evaluation:** (In Development) Compare different embedding models and retrieval strategies.
- **Vector Storage:** Integrates with [Qdrant](https://qdrant.tech/) for efficient similarity search.
- **Incremental Tracking:** Detects new or modified files to avoid redundant processing.
- **Streamlit UI:** Simple web interface for uploading documents and viewing recommendations.

## Project Structure

- `app.py`: Streamlit web interface for file uploads.
- `ingest.py`: Core ingestion logic for processing PDFs and updating the vector store.
- `data/pdfs/`: Local directory for source PDF documents.
- `storage/`: Local storage for index persistence (in development).
- `docker-compose.yml`: Configuration for running Qdrant locally.

## Getting Started

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker and Docker Compose

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd test-app
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Start Qdrant:
   ```bash
   docker-compose up -d
   ```

### Usage

#### Ingest Documents
To process the PDFs in the `data/pdfs` folder:
```bash
uv run python ingest.py
```

#### Run the UI
To start the Streamlit upload interface:
```bash
uv run streamlit run app.py
```

## Roadmap

- [ ] Implement robust configuration management (`config.toml`).
- [ ] Add support for multiple embedding providers.
- [ ] Implement advanced incremental tracking.
- [ ] Containerize the entire application.

## License

MIT
