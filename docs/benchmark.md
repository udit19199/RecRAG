# Dataset choices and source records

RecRAG uses four question-answer datasets. A **record** is one dataset item. It
contains one question, its answer, and the text supplied with that item. The
completed measured run uses one record from three datasets. The BrowseComp-Plus
run is pending.

The datasets store that text in different shapes:

| Dataset | Text stored in each record |
| --- | --- |
| HotpotQA | Sentences grouped under page titles |
| 2WikiMultiHopQA | Paragraphs grouped under page titles |
| Natural Questions | A complete Wikipedia page as HTML and tokens |
| BrowseComp-Plus | Evidence, gold, and hard-negative documents grouped under one query |

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

## Natural Questions

**Example ID:** `4549465242785278785`

**Question:** When is the last episode of season 8 of *The Walking Dead*?

**Answer:** March 18, 2018.

**Page:** *The Walking Dead (season 8)*

**Text around the answer:**

| Episode | Title | Date |
| ---: | --- | --- |
| 12 | "The Key" | March 18, 2018 |

The raw record stores the complete Wikipedia page in `document_html`. It also
stores a token list and the answer's location in that page. The `document_url`
identifies the Wikipedia revision used for the record. The page text is already
local, so the URL does not need to be fetched to read this example.

See the [Natural Questions data format](https://github.com/google-research-datasets/natural-questions#data-format).

# API cost and construction results

This section reports one-record construction experiments. Each completed
dataset used both graph-construction methods and then evaluated both graphs.
Construction did not receive the dataset question or answer. Only
`gpt-5.6-luna` was used for chat and evaluation, with Neo4j as the store.

Retrieval, retrieval evaluation, and answer generation were not run. Embedding
usage from `text-embedding-3-small` is not included in the LLM token totals
below.

## Price calculation

Prices are in US dollars per one million tokens. `M` means one million.

| Model | Input price | Output price | Source |
| --- | ---: | ---: | --- |
| `gpt-5.6-luna` | $0.20/M | $1.20/M | The `[cost]` section in `config.toml`. |

```text
cost = 0.20 × input tokens / M + 1.20 × output tokens / M
```

The token counts come from LangChain's `get_usage_metadata_callback()`. Costs
are rounded to six decimal places.

## Measured records

| Dataset | Record used | Pages or documents | Source characters | Status |
| --- | --- | ---: | ---: | --- |
| HotpotQA | `5a8b57f25542995d1e6f1371` | 10 | 4,430 | Complete |
| 2WikiMultiHopQA | `8813f87c0bdd11eba7f7acde48001122` | 10 | 3,578 | Complete |
| Natural Questions | `5225754983651766092` | 1 | 13,390 | Complete |
| BrowseComp-Plus | `query_id=775` | 78 | 710,164 | Pending: full-record construction did not complete |

The BrowseComp-Plus record contains six evidence documents, one gold document,
and 72 negative documents. Its full-record run did not produce a completed
usage measurement, so it is excluded from the totals below.

## Token usage and cost

| Dataset | Stage | Input tokens | Output tokens | Total tokens | Cost |
| --- | --- | ---: | ---: | ---: | ---: |
| HotpotQA | Standard construction | 17,536 | 6,082 | 23,618 | $0.010806 |
| HotpotQA | Ontology-guided construction | 13,471 | 5,032 | 18,503 | $0.008733 |
| HotpotQA | Standard construction evaluation | 6,568 | 459 | 7,027 | $0.001864 |
| HotpotQA | Ontology-guided construction evaluation | 5,506 | 482 | 5,988 | $0.001680 |
| 2WikiMultiHopQA | Standard construction | 18,124 | 5,514 | 23,638 | $0.010242 |
| 2WikiMultiHopQA | Ontology-guided construction | 11,263 | 5,684 | 16,947 | $0.009073 |
| 2WikiMultiHopQA | Standard construction evaluation | 5,969 | 743 | 6,712 | $0.002085 |
| 2WikiMultiHopQA | Ontology-guided construction evaluation | 5,918 | 530 | 6,448 | $0.001820 |
| Natural Questions | Standard construction | 42,025 | 8,632 | 50,657 | $0.018763 |
| Natural Questions | Ontology-guided construction | 34,083 | 12,386 | 46,469 | $0.021680 |
| Natural Questions | Standard construction evaluation | 10,524 | 402 | 10,926 | $0.002587 |
| Natural Questions | Ontology-guided construction evaluation | 12,213 | 382 | 12,595 | $0.002901 |

## Dataset totals

| Dataset | Input tokens | Output tokens | Total tokens | Cost |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 43,081 | 12,055 | 55,136 | $0.023082 |
| 2WikiMultiHopQA | 41,274 | 12,471 | 53,745 | $0.023220 |
| Natural Questions | 98,845 | 21,802 | 120,647 | $0.045931 |
| All three completed datasets | 183,200 | 46,328 | 229,528 | $0.092234 |

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
| Natural Questions | Standard | 31.923 | 98 | 70 | 7 | 2 | 36 | 0 | 0 |
| Natural Questions | Ontology-guided | 60.722 | 202 | 90 | 3 | 12 | 25 | 0 | 1 |

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
| Natural Questions | Standard | 0.2 | 0.3 | 0.6 |
| Natural Questions | Ontology-guided | 0.5 | 0.6 | 0.4 |

## Not yet measured

- Agentic retrieval, vector retrieval, and entity-vector retrieval.
- Retrieval evaluation and final answer generation.
- Embedding token usage and vector-storage cost.
- A completed BrowseComp-Plus construction and evaluation run.
