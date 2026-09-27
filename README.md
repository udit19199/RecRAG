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

4. Install Git LFS, then fetch the dataset files with `git lfs pull`.

5. Run the app:

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

## What the three methods do

All three answer the same dataset questions. They read `config.toml` for
`text-embedding-3-small` (1,536 dimensions) and `gpt-6-luna` (medium reasoning).
GPT-6 Luna handles answers and, where needed, graph construction through the
Responses API.

| Method | Before a question | How it finds evidence |
| --- | --- | --- |
| Normal RAG | Splits source pages into chunks and embeds each chunk. It does not build a graph. | LlamaIndex ranks chunks by vector similarity to the question. |
| HippoRAG2 | Embeds the same chunks and extracts entities and facts into its own local graph. | Its graph retrieval links question entities and facts to relevant chunks. |
| GraphRAG | Uses the ontology to build a graph and chunk vector index in Neo4j. | An agent chooses Neo4j vector search or a read-only text-to-Cypher query, then passes the evidence to the answer model. |

Normal RAG and HippoRAG2 use the shared baseline runner below. They use
256-token chunks with 20-token overlap and retrieve ten chunks. GraphRAG is the
app's separate research method: it currently uses 1,000-character chunks with
100-character overlap and returns at most five items. Keep these differences
with the results when comparing methods.

## RAG and HippoRAG baselines

The official HippoRAG repository is a separate, gitignored checkout at
`hipporag`. To create it on another machine:

```sh
gh repo clone OSU-NLP-Group/HippoRAG hipporag
```

Run both retrieval methods through [hipporag_baseline.py](hipporag_baseline.py).
Its dependencies conflict with Neo4j GraphRAG, so install them separately:

```sh
uv venv --python 3.12 .venv-hipporag
uv pip install --python .venv-hipporag/bin/python -e ./hipporag datasets huggingface-hub python-dotenv 'llama-index-core==0.14.23'
HF_HOME="$PWD/runs/hipporag-cache" .venv-hipporag/bin/python hipporag_baseline.py --method rag --dataset hotpotqa --records 1
HF_HOME="$PWD/runs/hipporag-cache" .venv-hipporag/bin/python hipporag_baseline.py --method hipporag --dataset hotpotqa --records 1
```

The runner reads `.env` and `config.toml`. Both methods use the configured
`text-embedding-3-small` with 1,536 dimensions. HippoRAG2 extraction and fact
filtering, and both methods' answers, use GPT-6 Luna with medium reasoning
through the Responses API. Both baselines store their indexes locally and need
neither Neo4j nor Milvus.

Choose `hotpotqa`, `multihop_rag`, or `natural_questions` with `--dataset`.
Use `--start` and `--records` to select records. Both methods use the same
source pages, including the full MultiHop-RAG corpus. LlamaIndex's
`SentenceSplitter` makes 256-token chunks with 20-token overlap; each chunk
retains its page title. Both retrievers return ten chunks to the same answer
prompt. Dense RAG uses LlamaIndex's vector index; HippoRAG uses its native
graph retrieval. These choices follow the paper's
[RAG](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/main/dataset.py) and
[HippoRAG](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/main/hippo_dataset.py)
indexing paths.

Each invocation saves settings, indexes, and per-record `results.jsonl` under
`runs/rag/` or `runs/hipporag/`, plus a per-dataset `summary.json`. Results
include retrieved text and scores,
answer precision, recall, F1, exact match, the paper's permissive MultiHop-RAG
overlap accuracy, answer-string retrieval accuracy, token counts, separate
timings, storage size, and LLM usage for uncached calls. Scores are deterministic
and use the same calculation for both methods. The paper's missing NQ/HotpotQA
scoring helper prevents exact score parity. Retrieved-token counts use
`cl100k_base`; the paper does not specify its counting tokenizer.

The paper's HotpotQA results use an unpublished random sample of 1,000 hard
bridge questions. This runner uses the official validation split in record
order, so its HotpotQA sample is different. The authors' processed document
reader and extraction helpers are also unavailable; both local methods use
the same titled source pages instead.

The upstream checkout stays unchanged. The paper used `text-embedding-ada-002`,
GPT-4o-mini for HippoRAG construction, and Llama-3.1-8B-Instruct for answers.
These runs instead use the shared RecRAG models above, so the paper's scores
cannot be compared directly with them. The paper also does not pin its
HippoRAG commit or publish its processed input records.

## Read the docs

- [Dataset and cost notes](docs/benchmark.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluation](docs/graphrag/evaluation.md)
- [Router](docs/router.md)
