# RecRAG

RecRAG is a research project by HPE and JECRC University. It supports HotpotQA,
MultiHop-RAG, Natural Questions, and NovelQA.

## Run the app

1. Install the dependencies:

   ```sh
   uv sync
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Install Git LFS, then fetch the dataset files with `git lfs pull`.

5. Run the app:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

## Prepare benchmark data

The app loads official validation/dev splits for these datasets on demand:
[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG), and
[Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions).
NovelQA is available from its
[gated Hugging Face dataset](https://huggingface.co/datasets/NovelQA/NovelQA).
For HotpotQA, the loader uses
`datasets/hotpotqa/hotpotqa_distractor_validation_first_1000.json` when present
(the first 1,000 source-order rows of the `distractor` validation split);
otherwise it loads the full split from Hugging Face.
Install the loader dependencies, then select records in the app:

```sh
uv run --with datasets --with huggingface-hub streamlit run streamlit_app.py
```

For each HotpotQA run, the GraphRAG app builds one shared graph per construction
method from unique pages across the loaded validation rows. The record count
selects questions to answer; it does not limit the graph's source pages.

HotpotQA loads the official `distractor` validation split. MultiHop-RAG loads
the official raw `MultiHopRAG.json` queries and `corpus.json` news corpus,
matching the filenames read by the reference implementation. Natural Questions loads
the official validation split (the original dev set). NovelQA loads
public-domain books and question files after you accept the dataset's access
terms and sign in to Hugging Face. The publisher keeps full-set answers and
evidence private to prevent answer leakage. A separate demonstration subset has
gold labels, but this adapter loads the full-set inputs, so its runs have no QA
scores. Remote data stays in the Hugging Face cache; the local HotpotQA sample
is ignored by Git.

## What the four methods do

All four answer the same dataset questions. They read `config.toml` for
`text-embedding-3-small` (1,536 dimensions) and `gpt-6-luna` (medium reasoning).
GPT-6 Luna handles answers and, where needed, graph construction through the
Responses API.

| Method | Before a question | How it finds evidence |
| --- | --- | --- |
| Normal RAG | Splits source pages into chunks and embeds each chunk. It does not build a graph. | LlamaIndex ranks chunks by vector similarity to the question. |
| RAPTOR | Builds a summary tree using global and local UMAP reduction, soft GMM clusters, and recursive summaries. | Ranks chunks and summaries together by cosine similarity, with a 3,500-token context limit. |
| HippoRAG2 | Embeds the same chunks and extracts entities and facts into its own local graph. | Its graph retrieval links question entities and facts to relevant chunks. |
| GraphRAG | Uses the ontology to build a graph and chunk vector index in Neo4j. | An agent chooses Neo4j vector search or a read-only text-to-Cypher query, then passes the evidence to the answer model. |

Normal RAG and HippoRAG2 use the shared baseline runner below. They use
256-token chunks with 20-token overlap and retrieve ten chunks. GraphRAG is the
app's separate research method: it currently uses 1,000-character chunks with
100-character overlap and returns at most five items. Keep these differences
with the results when comparing methods.

## Comparison runner

HippoRAG installs from its official GitHub repository at a fixed commit. Its
source files are not in the RecRAG working tree.

Normal RAG, RAPTOR, HippoRAG, and the ontology agentic GraphRAG method share
dataset preparation, benchmark prompts, metrics, and question cost measurement. Each owns its model clients and retrieval setup in `rag_retrieval.py`
and `hipporag_retrieval.py`. Normal RAG uses LlamaIndex and OpenAI directly,
without importing HippoRAG or GraphRAG. HippoRAG's pinned
dependencies conflict with newer LlamaIndex and Neo4j GraphRAG versions, so use
a separate environment:

```sh
uv venv --python 3.12 .venv-hipporag
uv pip install --python .venv-hipporag/bin/python \
  'hipporag @ git+https://github.com/OSU-NLP-Group/HippoRAG.git@1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff' \
  datasets huggingface-hub python-dotenv 'llama-index-core==0.11.23'
HF_HOME="$PWD/runs/hipporag-cache" \
  .venv-hipporag/bin/python -m hipporag_retrieval --dataset hotpotqa --records 1
```

Run Normal RAG without the HippoRAG environment:

```sh
uv run --with llama-index-core --with datasets --with huggingface-hub \
  python -m rag_retrieval --dataset hotpotqa --records 1
```

LlamaIndex Core 0.11.23 works with HippoRAG's Pydantic pin. The package install
uses the same upstream commit that RecRAG used from its local checkout.

The runner reads `.env` and `config.toml`. Both methods use the configured
`text-embedding-3-small` with 1,536 dimensions. HippoRAG2 extraction and fact
filtering, and all methods' answers, use GPT-6 Luna with medium reasoning
through the Responses API. Dense RAG, RAPTOR, and HippoRAG store their indexes locally and need
neither Neo4j nor Milvus.

Choose `hotpotqa`, `multihop_rag`, `natural_questions`, or `novelqa` with
`--dataset`.
Use `--start` and `--records` to select questions. For HotpotQA, both methods
search the unique context pages from the loaded validation rows. The flags select
which questions to answer, not which pages to search. This is the distractor
split, not all of Wikipedia. HippoRAG saves this index and reuses it in later
runs. For other datasets, the existing corpus behavior is retained. LlamaIndex's
`SentenceSplitter` makes 256-token chunks with 20-token overlap; each chunk
retains its page title. The baseline methods request ten items for the same
answer prompt. IRCoT can return more items because it keeps the union across
searches. Dense RAG uses LlamaIndex's vector index; HippoRAG uses its native
graph retrieval. These choices follow the paper's
[RAG](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/main/dataset.py) and
[HippoRAG](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/main/hippo_dataset.py)
indexing paths.

NovelQA uses only public-domain book text in its Hugging Face repository. Its
full-set question files omit the gold answers, so the runner saves answers with
null score fields for this dataset. The publisher's separate demonstration
subset has gold labels; full-set labels require a request to the publisher.

Each invocation first saves all retrieved text and question details in
`retrieval.jsonl`. A shared answer step reads that file and generates answers.
It scores records when gold answers exist. All methods use the same GPT-6 Luna
answer client.
To repeat answering without rebuilding an index or searching again:

```sh
uv run --with llama-index-core python -m benchmark \
  --answer-from runs/rag/<run-id>/retrieval.jsonl
```

This accepts retrieval files from all four methods. It uses the saved run settings and
replaces that run's answers and summary. The saved retrieval stays unchanged.

Each invocation saves settings, per-record `results.jsonl`, and a `summary.json`
under `runs/<method>/`. HippoRAG's HotpotQA index is saved once
under `runs/hipporag/indexes/` and reused by later runs. Results include
retrieved text, scores when available, token counts, timings, storage size, and
LLM usage for question-time calls. NovelQA score fields are null unless gold
labels are supplied. Metrics include
answer precision, recall, F1, exact match, the paper's permissive MultiHop-RAG
overlap accuracy, and answer-string retrieval accuracy. Scores are deterministic
and use the same calculation for all methods. The paper's missing NQ/HotpotQA
scoring helper prevents exact score parity. Retrieved-token counts use
`cl100k_base`; the paper does not specify its counting tokenizer.

The paper's HotpotQA results use an unpublished random sample of 1,000 hard
bridge questions. This runner uses the official validation split in record
order (or its first 1,000 rows when the local sample is present), with the
requested `--start` and `--records` range, so its sample is
different and does not specifically select hard bridge questions. Each query
searches unique context pages from the loaded validation rows. The run's
`corpus.json` records the question range, page count, chunk count, and corpus
hash. The paper's exact sample and processed files remain unavailable.

HotpotQA answer precision, recall, and F1 use the official evaluator's
normalization, token overlap, and yes/no handling. The paper's evaluation code
calls `calculate_metrics`, but that helper is not included in its linked repo,
so we cannot verify exact formula parity. The paper reports per-question
precision, recall, and F1 averaged across questions. The paper's protocol saves
retrieved evidence, then applies one generation script with Llama 3.1 8B or
70B. RecRAG uses `gpt-6-luna`, and its agentic ontology GraphRAG is not the
paper's Microsoft Community-GraphRAG. Our score formula follows HotpotQA's
official scorer, but exact parity with the paper's helper is unverified.

For paper comparisons, report HotpotQA precision, recall, and F1, and
MultiHop-RAG `overlap_accuracy`. Exact match is an additional diagnostic,
not the paper's headline metric. Both QA scorers extract the authors' answer
wrapper `The answer to the question is "..."` before scoring. MultiHop-RAG
matches the published evaluator: trim the prediction, remove its periods,
lowercase prediction and reference, and accept any shared whitespace-delimited
word. It does not require exact yes/no answers or use an LLM judge.

The upstream HippoRAG source stays unchanged. The paper used
`text-embedding-ada-002`,
GPT-4o-mini for HippoRAG construction, and Llama-3.1-8B-Instruct for answers.
These runs instead use the shared RecRAG models above, so the paper's scores
cannot be compared directly with them. The paper also does not pin its
HippoRAG commit or publish its processed input records.

## RAPTOR, reranking, IRCoT, and question cost

Install the research dependencies into the main environment. These are optional
benchmark tools, so the app dependency lockfile stays unchanged:

```sh
uv pip install --python .venv/bin/python llama-index-core datasets huggingface-hub \
  scikit-learn umap-learn sentence-transformers
```

Run the same question range with `--method rag`, `raptor`, or `graphrag`:

```sh
HF_HOME="$PWD/runs/hf-cache" .venv/bin/python -m benchmark \
  --method raptor --dataset multihop_rag --records 10
HF_HOME="$PWD/runs/hf-cache" .venv/bin/python -m benchmark \
  --method rag --dataset natural_questions --records 10 --rerank
HF_HOME="$PWD/runs/hf-cache" .venv/bin/python -m benchmark \
  --method raptor --dataset novelqa --records 10 --ircot --max-steps 2
HF_HOME="$PWD/runs/hf-cache" .venv/bin/python -m benchmark \
  --method graphrag --dataset multihop_rag --records 10
```

Use the existing separate environment for `--method hipporag`. Install
`sentence-transformers` there if you use `--rerank`. GraphRAG requires Neo4j
Desktop with APOC. The command runner caches one graph for each corpus and
configuration. It uses the existing ontology and agentic Neo4j tools, then
passes their evidence to the shared answer step. Its chunks remain 1,000
characters with 100-character overlap. Baseline chunks are 256 tokens with
20-token overlap. Normal retrieval returns ten items by default; reranking
searches 20 candidates before keeping ten. The app keeps its existing five-item
limit.

`--rerank` retrieves 20 candidates, scores them with the paper's
`BAAI/bge-reranker-large` cross-encoder, and keeps the top ten. Change those
limits with `--rerank-candidates` and `--rerank-topk`. `--ircot` uses the
paper's intermediate reasoning prompt, then searches again and stops on
`So the answer is:`. It deduplicates the union of evidence by whitespace-normalized
text. The default is two reasoning steps. The configured Responses model
generates the thoughts. As in upstream code, each nonfinal step triggers another
retrieval, including the last allowed step. The final answer is generated
separately from the evidence. Run reranking and IRCoT separately, matching the
paper's reported comparisons.

RAPTOR follows the authors' five-layer tree, ten-dimensional UMAP, soft GMM
membership threshold of 0.1, BIC cluster selection, 6,000-token cluster budget,
and collapsed-tree search. It stops construction when a layer has at most eleven
nodes. Trees are saved as JSON rather than pickle. UMAP uses the upstream
default randomness. Cluster membership uses only the upstream probability
threshold, without forced assignments. Oversized clusters recurse once per
level; if reclustering makes no progress, the run fails instead of splitting
by source order. GPT-6 Luna summaries request 100 tokens and retain the full
returned text, without cutting a sentence after generation. The configured
model substitutions prevent exact numerical reproduction of the paper.

Each result includes `question_cost`, an API-call ledger, and
`estimated_question_cost_usd`. This covers query embeddings, retrieval model
calls, HippoRAG filtering, IRCoT, and final answering. Index construction is
excluded even on a cold cache. The summary reports total and mean question API
cost and mean retrieval-plus-answer latency. Local reranking time counts toward
latency; local compute, database hosting, and storage costs are excluded.
`config.toml` stores the price snapshot, which each run copies to its settings.
Costs use API-reported input, cached-input, cache-write, output, and reasoning
counts. Reasoning tokens are already in output usage and are not charged twice.
Missing usage, an unpriced model, or old retrieval without a cost ledger produces
null cost instead of a false zero. Answer replay combines original retrieval
usage with the new answer call; it does not rerun retrieval. This is an API cost
estimate, not an invoice, and assumes the direct API without a regional premium.

The GraphRAG app saves the same question cost fields and displays an API cost
metric. Python callers of `GraphRAG.answer()` can read `rag.last_question_cost`
after each question without changing the answer result shape.

For input formats, score limits, and the exact upstream references, see
[the paper comparison notes](docs/benchmark.md#paper-comparison-protocol).

## Read the docs

- [Dataset and cost notes](docs/benchmark.md)
- [Graph construction](docs/graphrag/construction.md)
- [Graph retrieval](docs/graphrag/retrieval.md)
- [GraphRAG evaluation](docs/graphrag/evaluation.md)

## Verification

The repository does not use test files or test frameworks. Inspect live inputs
and saved output records, then run these static checks:

```sh
uv pip install --python .venv/bin/python ruff pyright
.venv/bin/python -m compileall -q benchmark.py cost.py dataset.py \
  rag_retrieval.py raptor_retrieval.py hipporag_retrieval.py graphrag_retrieval.py \
  streamlit_app.py evals
.venv/bin/ruff check benchmark.py cost.py dataset.py rag_retrieval.py \
  raptor_retrieval.py hipporag_retrieval.py graphrag_retrieval.py streamlit_app.py evals/qa_metrics.py
.venv/bin/ruff format --check benchmark.py cost.py dataset.py rag_retrieval.py \
  raptor_retrieval.py hipporag_retrieval.py graphrag_retrieval.py streamlit_app.py evals/qa_metrics.py
.venv/bin/pyright --pythonpath "$PWD/.venv/bin/python" benchmark.py cost.py dataset.py \
  rag_retrieval.py raptor_retrieval.py graphrag_retrieval.py streamlit_app.py evals/qa_metrics.py
.venv/bin/pyright --pythonpath "$PWD/.venv-hipporag/bin/python" hipporag_retrieval.py
```
