# RecRAG

RecRAG is a research project by HPE and JECRC University. It compares graph
construction and retrieval methods on four QA datasets, and studies a router
that maps a natural-language RAG use case to a suitable combination.

## Run the app

1. Install the dependencies:

   ```sh
   uv sync --extra experiments
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Install Git LFS, then fetch the dataset files with `git lfs pull`.

5. Prepare the local Natural Questions and BrowseComp-Plus files:

   ```sh
   uv run python scripts/setup_datasets.py
   ```

   The Natural Questions shards were reserialized from the [full validation split](https://huggingface.co/datasets/rongzhangibm/NaturalQuestionsV2).
   BrowseComp-Plus stays obfuscated in Git. The command creates its plaintext
   loader file locally, and `.gitignore` keeps that file out of commits.

6. Run the app:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

## Read the docs

- [Dataset and cost notes](docs/benchmark.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluation](docs/graphrag/evaluation.md)
- [Router](docs/router.md)
