Phase 1: Working Pipeline

# Step 1: Create Pipeline Modules (Flat Structure)
## Create these files in project root:
- ```config_loader.py``` - Load and parse config.toml with env var substitution
- embeddings.py - Embedding provider interface + OpenAI implementation
- BaseEmbedder class
- OpenAIEmbedder class
- Factory function to create embedder from config
qdrant_store.py - Qdrant integration
- Connect to Qdrant
- Create collection per provider (e.g., openai_text-embedding-3-small)
- Store vectors with metadata (filename, chunk_index, modified_time)
tracker.py - Incremental ingestion tracking
- Track: filename → {last_modified, chunks_count, indexed_at}
- Save/load from state/indexed_files.json
- Detect new/changed files by comparing mtime
pipeline.py - Main ingestion orchestrator
- Load config
- Find new/changed PDFs
- Load documents with SimpleDirectoryReader
- Chunk with 1024 tokens, 50 overlap
- Generate embeddings
- Store in Qdrant with metadata
- Update tracker
# Step 2: Rewrite ingest.py
Replace current content to:
- Import and use pipeline
- Accept CLI args (--config, --force)
- Run pipeline and report results
# Step 3: Test End-to-End
- Start Qdrant
```docker-compose up -d```
- Run ingestion
```uv run python ingest.py```
- Verify in Qdrant UI: http://localhost:6333/dashboard
- Test incremental: Add new PDF, run again
---
Phase 2: Add More Embedding Providers
- Implement HuggingFaceEmbedder in embeddings.py
- Implement OllamaEmbedder in embeddings.py
- Update config.toml to support provider-specific options
---
Phase 3: Production Polish
- Error handling and retry logic
- Logging with progress tracking
- Integration with app.py (Re-index button)
- CLI improvements
---
Phase 4: Containerization
- Dockerfile for the app
- Update docker-compose.yml to include app service
- Production config management
---
