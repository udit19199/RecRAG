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
2. Restart containers: `docker-compose down && docker-compose up -d`

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
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### No documents found for querying

- Upload your full PDF batch before querying
- Wait for status to show "complete"
- Check logs: `docker-compose logs api-ingestion`

### Permission errors

```bash
chmod 777 storage/ data/
```

### Container won't start

```bash
docker-compose build
docker-compose logs
```
