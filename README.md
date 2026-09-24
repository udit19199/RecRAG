# RecRAG

RecRAG is a research project by HPE and JECRC University. Its main benchmarks
are HotpotQA and MultiHop-RAG, with query-type analysis on MultiHop-RAG.

## Run the app

1. Install the dependencies:

   ```sh
   uv sync --extra experiments
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Start Milvus with `docker compose up -d`.

5. Install Git LFS, then fetch the dataset files with `git lfs pull`.

6. Run the app:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

## Prepare benchmark data

The app loads the official validation/dev splits on demand with the publishers'
Hugging Face dataset loaders: [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG), and
[Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions).
Install the loader dependencies, then select records in the app:

```sh
uv run --with datasets --with huggingface-hub streamlit run streamlit_app.py
```

HotpotQA loads the official `distractor` validation split. MultiHop-RAG loads
the official query split and its 609-document corpus. Natural Questions loads
the official validation split (the original dev set). Data is cached by the
Hugging Face datasets library; it is not copied into the repository.

HippoRAG 2 has a separate dependency stack. Create an isolated environment
before installing it so its pinned dependencies do not change RecRAG's lockfile:

```sh
uv venv --python 3.10 .venv-hipporag
uv pip install --python .venv-hipporag/bin/python hipporag==2.0.0a5
```

The [current official HippoRAG OpenAI adapter](https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/hipporag/llm/openai_gpt.py)
uses Chat Completions. RecRAG requires GPT-6 Luna through the Responses API, so
HippoRAG needs a compatible Responses adapter before it can be included in a
comparable run.

## Read the docs

- [Dataset and cost notes](docs/benchmark.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluation](docs/graphrag/evaluation.md)
- [Router](docs/router.md)
