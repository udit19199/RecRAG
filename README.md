# RecRAG

RecRAG is a research project by HPE and JECRC University. It compares graph
construction and retrieval methods on 2WikiMultiHopQA.

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

- [Research overview](docs/README.md)
- [Cost and model estimates](docs/api-cost-estimate.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluations](docs/graphrag/evals.md)
- [2WikiMultiHopQA results](docs/2wikimultihop.md)
