# Paper comparison protocol

The implementation reference is [RAGvsGraphRAG at commit
`d2a0c0c0deb0903d60338d3c416ccd6f9544267c`](https://github.com/haoyuhan1/RAGvsGraphRAG/tree/d2a0c0c0deb0903d60338d3c416ccd6f9544267c).
The ACM PDF returned HTTP 403 during this implementation. The authors'
[public paper, arXiv v3](https://arxiv.org/html/2502.11371v3), provides the
method and dataset protocol.

| Dataset | Index scope | Answers and scores |
| --- | --- | --- |
| MultiHop-RAG | One index over the complete news corpus for every question | Short answers, binary answers, temporal answers, and insufficient-information answers. Preserve `inference_query`, `comparison_query`, `temporal_query`, and `null_query`; report overlap accuracy for each type. |
| Natural Questions | One index for each source Wikipedia document, reused by its questions | Answer aliases with precision, recall, F1, and exact match. |
| NovelQA | One index for each novel, reused by its questions | Multiple-choice letters. Report exact letter accuracy when gold labels are supplied. Preserve aspect and complexity. |

These scopes apply to dense RAG, HippoRAG2, RAPTOR, and RecRAG's ontology
agentic GraphRAG comparison. The official-source loaders remain the default.
Indexes are keyed by source chunks, method, model settings, and algorithm
version, so later question ranges reuse an index without mixing models.

The authors' repository does not publish `dataset/NQ.json`,
`dataset/NovelQA_contexts.json`, `dataset/NovelQA_qa.json`, `utils.JSONReader`,
or its metadata extractors. Their precise sample and undocumented context
serialization cannot be recovered from that repository. RecRAG's default NQ
validation records and NovelQA public-domain novels are therefore alternative
inputs, not the authors' exact processed collection. Full NovelQA labels
remain unavailable from the public full-set inputs.

If you obtain the processed files, keep them inside the repository under an
ignored dataset directory and pass `--paper-data datasets/paper`:

```sh
.venv/bin/python -m benchmark --method raptor --dataset natural_questions \
  --paper-data datasets/paper --records 10
```

The adapter accepts these explicit formats:

- `corpus.json` and `MultiHopRAG.json` use the official MultiHop-RAG fields.
- `NQ.json` is a list of documents with `questions` and `answer` arrays of equal
  length, and source text under `context`, `content`, or `text`. Each answer can
  be a string or a list of aliases. Context can be text or a list of passages.
- `NovelQA_contexts.json` maps book IDs or book filenames to text or passage
  lists. `NovelQA_qa.json` maps book IDs to question IDs, each with `Question`,
  `Options`, optional `Gold`, `Aspect`, and `Complexity`. Gold letters enable
  scores. Unsupported context shapes fail with a message rather than silently
  indexing an empty or wrong document.

The NQ and NovelQA context formats above are RecRAG's documented import
contract. They cannot be asserted to match the missing upstream reader until
real processed inputs are available. Gold labels and question options are
never used for indexing.

MultiHop-RAG reads the official raw `MultiHopRAG.json` and `corpus.json`
files directly, preserving query order and the complete shared news corpus.
The reference uses these filenames too. Its unpublished `JSONReader` and
`NewsExtractor` still prevent verification of exact text and metadata formatting.
RecRAG includes title, source, and publication date in each news page title.

MultiHop-RAG query types stay attached to each record and result. The loader
rejects labels outside the dataset's four published values so per-type results
cannot silently drop or misclassify a query.

RAPTOR's algorithm comes from [raptor.py](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/d2a0c0c0deb0903d60338d3c416ccd6f9544267c/raptor.py).
Reranking and IRCoT follow the [retrieval scripts](https://github.com/haoyuhan1/RAGvsGraphRAG/blob/d2a0c0c0deb0903d60338d3c416ccd6f9544267c/retrieval.py).
See the [run instructions](../README.md#raptor-reranking-ircot-and-question-cost)
for algorithm limits and known deviations. All methods use the configured
GPT-6 Luna and embedding model, so compare their results within RecRAG.
The paper's model choices and missing evaluation helper prevent exact score
reproduction. NovelQA exact match follows the upstream first-letter rule. The summary also
groups accuracy by aspect and complexity, and preserves both on every record.

The runner records retrieval and answer time and estimates question API cost
from API-reported usage. [Official OpenAI prices](https://developers.openai.com/api/docs/pricing)
were checked on 2026-10-04 and stored in `config.toml`. Indexing, evaluation,
and local machine costs are excluded. Both cached and cold indexes use the same
question-only cost boundary. The HippoRAG Responses adapter requests JSON
objects for extraction because the upstream NER list instruction conflicts with
its object parser. Its cache identity includes this adapter version. No new dataset benchmark results are claimed by
this implementation. Method routing remains a separate research comparison.

# Dataset choices and source records

RecRAG currently supports HotpotQA, MultiHop-RAG, Natural Questions, and
NovelQA. A **record** is one dataset item. It contains one question and the text
supplied with that item. Public NovelQA files do not include gold answers, so
its full-set runs have no answer scores. The publisher provides a separate
labeled demonstration and requires a request for full labels. A previous
BrowseComp-Plus trial is retained below as historical measurement context; it
is no longer an active adapter.

The datasets store that text in different shapes:

| Dataset | Text stored in each record |
| --- | --- |
| HotpotQA | Sentences grouped under page titles |
| MultiHop-RAG | News documents from a shared 609-document corpus |
| Natural Questions | The full Wikipedia article attached to the question |
| NovelQA | The public-domain novel attached to its questions |

Compare results within each dataset. The sources use different document
collection sizes and answer formats.

The examples below show the dataset formats in a readable form. The exact
records used for the measured run are listed later.

## HotpotQA

**Record ID:** `5a8b57f25542995d1e6f1371`

**Question:** Were Scott Derrickson and Ed Wood of the same nationality?

**Answer:** Yes.

**Text stored in the record:**

**Scott Derrickson**

> Scott Derrickson (born July 16, 1966) is an American director, screenwriter,
> and producer.
>
> He lives in Los Angeles, California.
>
> He is best known for directing horror films such as "Sinister", "The Exorcism
> of Emily Rose", and "Deliver Us From Evil", as well as the 2016 Marvel
> Cinematic Universe installment, "Doctor Strange."

**Ed Wood**

> Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American
> filmmaker, actor, writer, producer, and director.

The record marks the first sentence for both pages as supporting evidence. Both
people are described as American, so the answer is "yes".

## 2WikiMultiHopQA

**Record ID:** `8813f87c0bdd11eba7f7acde48001122`

**Question:** Who is the mother of the director of the film *Polish-Russian War*?

**Answer:** Małgorzata Braunek.

**Text stored in the record:**

**Polish-Russian War (film)**

> Polish-Russian War (Wojna polsko-ruska) is a 2009 Polish film directed by
> Xawery Żuławski based on the novel Polish-Russian War under the white-red flag
> by Dorota Masłowska.

**Xawery Żuławski**

> Xawery Żuławski (born 22 December 1971 in Warsaw) is a Polish film director.
> He is the son of actress Małgorzata Braunek and director Andrzej Żuławski.

**How the answer is found:**

1. The film's director is Xawery Żuławski.
2. Xawery Żuławski's mother is Małgorzata Braunek.

Here, `context` only groups the text by page. The `evidences` field stores the
same two links in a compact form: film → director, then director → mother.

# Historical API cost and benchmark results

The two completed datasets each use one record in this pilot. The run was
intended to estimate token use and cost, not to select a benchmark or rank
methods. All measured calls used `gpt-5.6-luna`; future runs will use
`gpt-6-luna`.

Construction usage comes from run `20260922T120012971066Z`; retrieval and answer
usage comes from `20260922224208763174`. The retrieval graphs were rebuilt, but
their construction usage was not captured. Token counts below are observed
`gpt-5.6-luna` counts; dollar values are projections using current `gpt-6-luna`
rates, not actual GPT-6 charges. Embedding and storage costs are excluded.

## Price calculation

OpenAI Standard short-context prices (up to 272K input tokens), in US dollars
per one million tokens. The last column applies each model's rates to the same
recorded token and cache counts from this pilot.

| Model | Uncached input | Cached input | Cache write | Output | Same-usage estimate | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt-5.6-luna` | $0.20/M | $0.02/M | $0.25/M | $1.20/M | $0.466927 | [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| `gpt-6-luna` | $0.10/M | $0.01/M | $0.125/M | $0.50/M | $0.219268 | [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-6-luna) |

```text
uncached input = input - cached input - cache writes
cost = (0.10 × uncached input + 0.01 × cached input
      + 0.125 × cache writes + 0.50 × output) / M
```

Cache-read and cache-creation counts come from saved usage metadata. Cost
estimates are rounded to six decimal places. At the same usage, GPT-6 Luna is
about 53% cheaper ($0.247659 less) than GPT-5.6 Luna. This lower GPT cost also
applies to the GPT calls in the RLCD comparison; the absolute amount is small
for this pilot.

## Measured records

| Dataset | Record used | Pages or documents | Source characters | Status |
| --- | --- | ---: | ---: | --- |
| HotpotQA | `5a8b57f25542995d1e6f1371` | 10 | 4,430 | Complete |
| 2WikiMultiHopQA | `8813f87c0bdd11eba7f7acde48001122` | 10 | 3,578 | Complete |
| Historical BrowseComp-Plus attempt | `query_id=775` | 78 | 710,164 | Full-record construction did not complete |

The BrowseComp-Plus record contains six evidence documents, one gold document,
and 72 negative documents. Its full-record run did not produce a completed
usage measurement, so it is excluded from the totals below.

## Token usage and cost

| Dataset | Stage | Input tokens | Output tokens | Total tokens | Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | Standard construction | 17,536 | 6,082 | 23,618 | $0.004877 |
| HotpotQA | Ontology-guided construction | 13,471 | 5,032 | 18,503 | $0.004180 |
| HotpotQA | Standard construction evaluation | 6,568 | 459 | 7,027 | $0.001041 |
| HotpotQA | Ontology-guided construction evaluation | 5,506 | 482 | 5,988 | $0.000899 |
| HotpotQA | Retrieval and answer (8 runs) | 68,455 | 490 | 68,945 | $0.008789 |
| HotpotQA | Retrieval evaluation (8 runs) | 202,699 | 19,109 | 221,808 | $0.034116 |
| HotpotQA | Answer evaluation (8 runs) | 83,816 | 10,107 | 93,923 | $0.014976 |
| 2WikiMultiHopQA | Standard construction | 18,124 | 5,514 | 23,638 | $0.005003 |
| 2WikiMultiHopQA | Ontology-guided construction | 11,263 | 5,684 | 16,947 | $0.004233 |
| 2WikiMultiHopQA | Standard construction evaluation | 5,969 | 743 | 6,712 | $0.001108 |
| 2WikiMultiHopQA | Ontology-guided construction evaluation | 5,918 | 530 | 6,448 | $0.000995 |
| 2WikiMultiHopQA | Retrieval and answer (8 runs) | 72,970 | 1,032 | 74,002 | $0.009624 |
| 2WikiMultiHopQA | Retrieval evaluation (8 runs) | 216,323 | 25,439 | 241,762 | $0.039040 |
| 2WikiMultiHopQA | Answer evaluation (8 runs) | 87,686 | 10,230 | 97,916 | $0.015504 |

## Dataset totals

| Dataset | Input tokens | Output tokens | Total tokens | Cost |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 398,051 | 41,761 | 439,812 | $0.068878 |
| 2WikiMultiHopQA | 418,253 | 49,172 | 467,425 | $0.075506 |
| Both completed datasets | 816,304 | 90,933 | 907,237 | $0.144384 |

## Exploratory retrieval and answer scores

These are raw scores already saved from run `20260922224208763174`, generated
with `gpt-5.6-luna`; values are shown per record and rounded to three decimals.
This was a one-record-per-dataset cost pilot, not a decided benchmark. The
benchmark protocol has not been selected, so these values are not aggregated
or treated as final findings. No new judge calls were made to prepare this
table; defer further scoring until the benchmark is agreed. Every saved result
in this pilot has five items, so the new shared top-five cap does not change
these saved scores.

| Dataset | Construction | Retrieval | Context precision | Context recall | Context relevancy | Answer relevancy | Correctness | Faithfulness | Exact alias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | standard | vector | 1.000 | 1.000 | 0.241 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | standard | entity_vector | 1.000 | 0.500 | 0.375 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | standard | milvus_vector | 1.000 | 1.000 | 0.258 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | ontology_guided | vector | 1.000 | 1.000 | 0.290 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | ontology_guided | entity_vector | 1.000 | 0.500 | 0.280 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | ontology_guided | milvus_vector | 1.000 | 1.000 | 0.375 | 1.000 | 1.000 | 1.000 | No |
| 2wikimultihopqa | standard | vector | 0.639 | 1.000 | 0.308 | 1.000 | 1.000 | 1.000 | Yes |
| 2wikimultihopqa | standard | entity_vector | 0.000 | 0.333 | 0.156 | 1.000 | 1.000 | 1.000 | No |
| 2wikimultihopqa | standard | milvus_vector | 1.000 | 1.000 | 0.231 | 1.000 | 1.000 | 1.000 | Yes |
| 2wikimultihopqa | ontology_guided | vector | 0.639 | 1.000 | 0.175 | 0.667 | 1.000 | 1.000 | No |
| 2wikimultihopqa | ontology_guided | entity_vector | 1.000 | 0.000 | 0.139 | 1.000 | 1.000 | 1.000 | No |
| 2wikimultihopqa | ontology_guided | milvus_vector | 0.639 | 1.000 | 0.250 | 0.667 | 1.000 | 1.000 | No |
| hotpotqa | standard | agentic | 1.000 | 1.000 | 0.258 | 1.000 | 1.000 | 1.000 | No |
| hotpotqa | ontology_guided | agentic | 1.000 | 1.000 | 0.345 | 1.000 | 1.000 | 1.000 | No |
| 2wikimultihopqa | standard | agentic | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | Yes |
| 2wikimultihopqa | ontology_guided | agentic | 1.000 | 1.000 | 0.250 | 1.000 | 1.000 | 1.000 | No |

## Construction metrics

These metrics are for the constructed Neo4j graph. Duplicate counts are
reported separately from the entity count because they show repeated names or
nodes that may need graph consolidation.

| Dataset | Method | Time (s) | Entities | Relationships | Relation types | Duplicate name groups | Duplicate nodes | Isolated entities | Self-loops |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | Standard | 31.646 | 77 | 62 | 12 | 0 | 19 | 0 | 0 |
| HotpotQA | Ontology-guided | 16.393 | 78 | 33 | 5 | 1 | 2 | 0 | 0 |
| 2WikiMultiHopQA | Standard | 36.812 | 59 | 46 | 18 | 0 | 16 | 0 | 0 |
| 2WikiMultiHopQA | Ontology-guided | 23.560 | 69 | 38 | 8 | 2 | 4 | 0 | 0 |

## Construction evaluation scores

Scores range from 0 to 1 and come from the construction evaluation run.
These are LLM-judged proxies. The datasets do not provide gold entity-relation
triple sets, so exact graph precision, recall, and F1 are not computed.

| Dataset | Method | Groundedness | Completeness | Supporting-evidence coverage |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Standard | 0.5 | 0.5 | 0.3 |
| HotpotQA | Ontology-guided | 0.7 | 0.6 | 0.4 |
| 2WikiMultiHopQA | Standard | 0.5 | 0.6 | 0.7 |
| 2WikiMultiHopQA | Ontology-guided | 0.5 | 0.7 | 0.6 |

## Gaps in that historical pilot

- Retrieval and end-to-end latency.
- Embedding token usage and vector-storage cost.
- GPT-6-Luna usage and results; current GPT-6 costs are projections on earlier token counts.

Retrieval, retrieval evaluation, answer generation, and answer evaluation are
complete for all four retrieval methods on the four graphs. The retrieval run
contains eight results per dataset: four retrieval methods on each of two
construction methods.
