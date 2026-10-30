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

For each HotpotQA run, the GraphRAG app builds one shared graph per construction
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
| GraphRAG | Uses the ontology to build a graph and chunk vector index in Neo4j. | An agent chooses Neo4j vector search or a read-only text-to-Cypher query, then passes the evidence to the answer model. |

All four baseline methods use the same title-prefixed `SentenceSplitter`
chunks, with 256 tokens per chunk and 20 tokens of overlap. GraphRAG writes
those exact chunks to Neo4j and returns at most five items. Normal RAG,
HippoRAG2, and RAPTOR return ten items by default.

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
bridge questions. RecRAG's local
`hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json` uses a
reproducible sample from the official validation split. Its exact questions
still differ from the paper's unpublished sample. Run with `--records 1000` to
answer the full sample. Each query searches unique context pages across those
1,000 rows. The run's `corpus.json` records the question range, page count,
chunk count, and corpus hash.

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

## Retrieval quality

Retrieval is scored from saved `retrieval.jsonl` files, so no method reruns.
The headline metric comes from one LLM judge applied identically to all four
methods:

```sh
uv run python -m evals.retrieval_judge runs/<method>/<run> ...
```

Pass one run per method per dataset, including GraphRAG. Exact-string and
offset matching cannot score RAPTOR's generated summaries, because the saved
runs no longer have the tree that ties a summary to its source pages. Reading
content is the only way to treat every architecture the same.

**Primary: answer support.** Gold sentences mark one route to the answer.
HotpotQA labels come from each question's own ten paragraphs, but our corpus
pools pages from 1,000 questions, so the same links often appear on other
pages. GraphRAG's 1,000-character windows can also cut a gold sentence
midway. Gold-sentence recall therefore penalized methods that reached the
answer by another stated route. When we measured this, GraphRAG had the answer
in its text for 28 of the 41 HotpotQA questions where it missed a gold
sentence; normal RAG had it for 8 of 27. The support judge gets the question,
the gold answer, the gold sentences as a reference route, and the items. It
quotes a chain to the gold answer, by that route or any route the items
state, and marks the answer stated, only mentioned, or missing. Code scores
each question: 3 a complete verified chain with the answer stated (yes/no
answers need only the complete chain); 2 the answer stated; 1 the answer
mentioned, or part of a chain; 0 nothing. Support score = sum / (3 ×
questions) × 100. Full support rate = share of questions scoring 3.

**Secondary: gold-sentence recall**, scored as follows.

- Each gold sentence is split into claims once per question. The claims are
  frozen, so every method is judged against the same ones.
- For each method and question, `gpt-6-luna` marks every claim stated,
  missing, or contradicted. It must quote the retrieved text. The judge sees
  no method name and no question, and the items are shuffled.
- Code, not the judge, sets each gold sentence's integer score, using only
  quotes that appear verbatim in the cited item: 3 every claim stated; 2 the
  main claim stated; 1 the sentence's people or things mentioned; 0 otherwise.
  Any verified contradiction scores 0.
- A question takes its lowest gold sentence score, because a multi-hop
  question needs every hop.
- **Retrieval score** = sum of question scores / (3 × questions) × 100.
  **Full retrieval rate** = share of questions scoring 3. Budget views cut
  ranked methods to GraphRAG's per-question token volume. Intervals are paired
  bootstraps.

Verdicts are cached in `runs/retrieval_judge/cache.jsonl`, keyed by prompt
version, model, and payload. A rerun resumes and re-scoring is free. Bump
`PROMPT_VERSION` whenever a prompt or the scoring rule changes.
`evals/retrieval_quality.py` keeps the older offset metrics. The judge reuses
them only as a check: `disagreements.jsonl` lists gold sentences where the two
disagree, and `summary_credit.jsonl` lists credit earned through RAPTOR
summaries, for hand review.

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
`sentence-transformers` there if you use `--rerank`. GraphRAG requires Neo4j
Desktop with APOC. The command runner caches one graph for each corpus and
configuration. It uses the existing ontology and agentic Neo4j tools, then
passes their evidence to the shared answer step. GraphRAG uses the same
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
