# RecRAG

RecRAG is a research project by HPE and JECRC University. It supports HotpotQA,
MultiHop-RAG, Natural Questions, and NovelQA.

The benchmark protocol follows the paper
[RAG vs. GraphRAG: A Systematic Evaluation and Key Insights](https://dl.acm.org/doi/pdf/10.1145/3770855.3817575)
and its reference implementation,
[haoyuhan1/RAGvsGraphRAG](https://github.com/haoyuhan1/RAGvsGraphRAG).
Method diagrams, cost estimates, and results are in the
[Experiments Excalidraw scene](https://app.excalidraw.com/s/9pwrOc81bDT/5KJVp4wI32J).

## Run the app

1. Install the dependencies:

   ```sh
   uv sync
   ```

2. Create `.env` from `.env.example` and set the required API keys.

3. Start Neo4j Desktop Enterprise with APOC enabled.

4. Run the app:

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
`datasets/hotpotqa/hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json`
when present. It contains 1,000 questions labeled `hard` and `bridge`, sampled
with seed 42 from the full `distractor` validation split and kept in source
order. Regenerate it with:

```sh
uv run --with datasets --with huggingface-hub \
  python scripts/prepare_hotpotqa_hard_bridge_sample.py
```

Fetch the pinned MultiHop-RAG questions and all 609 corpus documents into the
local-only `datasets/multihop_rag/` directory with:

```sh
uv run --with huggingface-hub python -m scripts.prepare_multihop_rag_data
```

If this file is absent, the loader uses an older local sample when present,
then falls back to the full validation split.
Install the loader dependencies, then select records in the app:

```sh
uv run --with datasets --with huggingface-hub streamlit run streamlit_app.py
```

For each HotpotQA run, the Agentic Hybrid RAG app builds one shared graph per construction
method from unique pages across the loaded validation rows. The record count
selects questions to answer; it does not limit the graph's source pages.

HotpotQA loads the official `distractor` validation split. MultiHop-RAG loads
the official raw `MultiHopRAG.json` queries and `corpus.json` news corpus,
matching the filenames read by the reference implementation. Keep local copies
under `datasets/multihop_rag/`; the loader uses them when present and otherwise
downloads the pinned dataset revision
`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`. Natural Questions loads
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
| Agentic Hybrid RAG | Uses the ontology to build a graph and chunk vector index in Neo4j. | An agent runs Neo4j vector search, then one read-only text-to-Cypher lookup, and may search again (up to 5 model calls). It ends by listing the tagged results the answer needs; those go to the answer model first, and the best-scoring passages fill the rest. |

Agentic Hybrid RAG was called Agentic GraphRAG until 15 Nov 2026. It combines vector
search and graph lookups, hence the rename. Module, file and result-key names still say
`graphrag`.

All four baseline methods use the same title-prefixed `SentenceSplitter`
chunks, with 256 tokens per chunk and 20 tokens of overlap. Agentic Hybrid RAG writes
those exact chunks to Neo4j. In benchmark runs every method passes ten items to
the answer model; the Streamlit app passes five.

## Comparison runner

HippoRAG installs from its official GitHub repository at a fixed commit. Its
source files are not in the RecRAG working tree.

Normal RAG, RAPTOR, HippoRAG, and the Agentic Hybrid RAG method share
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

How these runs differ from the paper (its unpublished samples, scoring
helper and models) is in
[the paper comparison notes](docs/benchmark.md#differences-from-the-paper).

## How answers and retrieval are scored

Every method is scored from its saved `retrieval.jsonl` and `results.jsonl`, so
scoring never reruns a method. Claude Haiku is the only judge; judging makes no
OpenAI calls. Two measures are reported, always split by question type:

- **Answered right**: the answer matches a gold answer after lowercasing and
  removing punctuation and a, an, the. Otherwise a Claude Haiku subagent decides
  whether it names the same entity, value or yes/no. An "Insufficient
  Information" reply that does not match is wrong. Export the cases that need
  the judge with `uv run python -m evals.subagent_judge export OUT RUN_DIR...`.
- **Gold facts found**, at ≥50%, ≥75% and 100% of a question's facts. Claude
  Haiku splits each question's gold sentences into short facts once, so every
  method is judged on the same facts. Then, per method and question, it marks
  each fact stated, contradicted or missing, with a quote. A fact counts only
  when it is stated and code finds the quote in the retrieved text. Run the
  three stages from `evals/haiku_fact_judge.py`:

  ```sh
  uv run python -m evals.haiku_fact_judge claims OUT RUN_DIR...
  uv run python -m evals.haiku_fact_judge judge OUT RUN_DIR...
  uv run python -m evals.haiku_fact_judge score OUT
  ```

The latest run's judge prompts, audit and per-question verdicts are in
`runs/final500/judge/` and `runs/final500/judge_prompts.json`.

The previous run (`runs/heldout400/`) scored gold facts with the older GPT-6
Luna judge in `evals/retrieval_judge.py` (verdict cache in
`runs/retrieval_judge/cache.jsonl`). Its numbers keep that judge; new runs use
the Haiku judge above. Gold-sentence recall, answer support and the offset
metrics in `evals/retrieval_quality.py` are retired as reported measures.

A correct answer can also come from model knowledge rather than retrieval.
`evals/closed_book.py` answers the same questions with no context, using the
same model, reasoning effort, and scorers, and writes to `runs/closed_book/`:

```sh
uv run python -m evals.closed_book runs/<method>/<hotpotqa-run> runs/<method>/<multihop-run>
```

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
`sentence-transformers` there if you use `--rerank`. Agentic Hybrid RAG requires Neo4j
Desktop with APOC. The command runner caches one graph for each corpus and
configuration. It uses the existing ontology and agentic Neo4j tools, then
passes their evidence to the shared answer step. Agentic Hybrid RAG uses the same
title-prefixed 256-token `SentenceSplitter` chunks with 20-token overlap as the
other benchmark methods. Normal retrieval returns ten items by default;
reranking searches 20 candidates before keeping ten. The app keeps its
existing five-item limit.

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
metered separately: each run writes `build_cost.json` with the calls made while
building an index (empty when every index already existed, since cached
extraction results are not re-billed). The summary reports total and mean question API
cost and mean retrieval-plus-answer latency. Local reranking time counts toward
latency; local compute, database hosting, and storage costs are excluded.
`config.toml` stores the price snapshot, which each run copies to its settings.
Costs use API-reported input, cached-input, cache-write, output, and reasoning
counts. Reasoning tokens are already in output usage and are not charged twice.
Missing usage, an unpriced model, or old retrieval without a cost ledger produces
null cost instead of a false zero. Answer replay combines original retrieval
usage with the new answer call; it does not rerun retrieval. This is an API cost
estimate, not an invoice, and assumes the direct API without a regional premium.

Measurement rules, so the four methods are timed and priced the same way:

- `list_question_cost_usd` prices every call at the uncached list rate. Billed
  cost depends on prompt-cache hits, which depend on run order and reruns.
- Every call records its wall time; `model_seconds` is their sum, so latency
  splits into model time and everything else (database, local compute).
- Failed or timed-out calls are listed in `question_cost.failed_calls`. A
  question is tried twice; if both attempts fail it is saved with
  `retrieval_error` and scored wrong, and the cost of every attempt is kept.
  Three failed questions in a row stop the run.
- Each index is opened once per run and one unscored warm-up question runs
  before timing starts, so lazy index loading is not charged to a question.
- HippoRAG's question-time calls bypass its response cache; a cache hit would
  record no cost and no time.
- `RECRAG_CONCURRENCY` (recorded per result) states how many workers shared
  the API. Compare latency only between runs with the same value.
- `python -m evals.subagent_judge export` writes the judgments exact matching
  cannot make (non-exact answers, evidence without a source position) for a
  Claude Haiku subagent, so judging spends no OpenAI credits.

The Agentic Hybrid RAG app saves the same question cost fields and displays an API cost
metric. Python callers of `GraphRAG.answer()` can read `rag.last_question_cost`
after each question without changing the answer result shape.

For input formats, score limits, and the exact upstream references, see
[the paper comparison notes](docs/benchmark.md#paper-comparison-protocol).

## Run results and findings

Run results are committed under `runs/`; indexes and caches there are
gitignored. The slides in the
[Experiments scene](https://app.excalidraw.com/s/9pwrOc81bDT/5KJVp4wI32J) are
the reported results.

| Folder | What it holds |
| --- | --- |
| `runs/final500/` | The latest run: MultiHop-RAG, HotpotQA and Natural Questions, 500 questions each, 4 methods, 3 repeats. `SPEC.md`, `DECISIONS.md`, `run_registry.json` (which run folders belong to it), `report.json`. |
| `runs/heldout400/` | The previous run: 400 MultiHop-RAG questions, 3 repeats. See its `README.md`. |
| `runs/agentic_tune/`, `runs/nq100/` | Agentic Hybrid RAG tuning runs on MultiHop-RAG and the first 100 Natural Questions, before the latest run. Each has a `README.md`. |
| `runs/hotpotqa100/` | Per-type scores rebuilt for an older HotpotQA 100-question run whose run folders were deleted. |
| `runs/nograph/` | Graph ablation: Agentic Hybrid RAG with the graph switched off (`RECRAG_AGENT_NO_GRAPH=1`), on the latest run's 250 MultiHop-RAG comparison and temporal questions, 3 repeats. `SPEC.md`, `run_registry.json`, `slide_data.json` (slides 09-10), `graph_audit.json`. |
| `runs/rag/`, `runs/raptor/`, `runs/hipporag/`, `runs/graphrag/` | Per-method run folders, including the latest run's MultiHop-RAG runs and the graph-ablation runs. |

Findings from the latest run, under `runs/final500/`:

- `two_run_analysis/`: previous vs latest run per question type. Agentic
  Hybrid RAG's wrong "Insufficient Information" answers fell from 24.0% to 9.1%
  (comparison) and from 37.0% to 11.5% (temporal) once the agent picked its own
  evidence; Normal RAG did not move.
- `repeat_variation/`: why answers varied between repeats (script and JSON
  output only, no summary yet).
- `cypher_failures/`: of 4,569 graph queries the model wrote, 337 (7.4%) failed,
  mostly invalid Cypher or `toLower` on a list field, and 1,121 (24.5%) ran but
  found nothing. 328 of 4,500 answers had at least one failed query.

Graph ablation, under `runs/nograph/` (answered right, mean of 3 repeats, 125
questions per type):

| Type | Normal RAG | Agentic Hybrid RAG | Agentic, no graph |
| --- | --- | --- | --- |
| Comparison | 51.7% | 70.4% | 70.1% |
| Temporal | 47.7% | 76.5% | 74.9% |

Switching the graph off barely moves the agent: most of its gain over Normal
RAG comes from the agent loop, not the graph. Gold facts were not judged for
this run.

## Read the docs

- [Paper comparison and dataset notes](docs/benchmark.md)
- [Agentic Hybrid RAG: graph construction, retrieval and app scores](docs/agentic-hybrid-rag.md)
- [Research directions](docs/research-directions.md)

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
