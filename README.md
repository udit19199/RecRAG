# RecRAG

RecRAG is a research project by HPE and JECRC University. It compares graph
construction and retrieval methods on 2WikiMultiHopQA, and studies a router
that maps a natural-language RAG use case to a suitable combination.

## Run the app

1. Install the dependencies:

   ```sh
   uv sync --extra experiments
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Run the app:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

## Read the docs

- [Documentation index](docs/index.md)
- [Router study](docs/router/router-study.md)
- [Classical RAG](docs/rag/classical-rag.md)
- [Cost and model estimates](docs/api-cost-estimate.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluations](docs/graphrag/evals.md)
- [2WikiMultiHopQA results](docs/graphrag/2wikimultihop.md)
- [Agentic RAG](docs/agenticrag/agentic-rag.md)
