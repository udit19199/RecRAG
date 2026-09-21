# Dataset choices and source records

RecRAG uses three question-answer datasets. A **record** is one dataset item. It
contains one question, its answer, and text from one or more Wikipedia pages.

The datasets store that text in different shapes:

| Dataset | Text stored in each record |
| --- | --- |
| HotpotQA | Sentences grouped under page titles |
| 2WikiMultiHopQA | Paragraphs grouped under page titles |
| Natural Questions | A complete Wikipedia page as HTML and tokens |

The examples below show the same records in a readable form.

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

# API cost estimate

This page gives the cost table for one record from each dataset. It covers graph
construction, retrieval, evaluation, and answer generation.

These are planning estimates. The datasets do not have a fixed record size. A
HotpotQA record can have a different number of passages from the next record,
and a Natural Questions record contains a full Wikipedia page. Replace the
variables in a table with the token counts returned by the provider for the
record under study.

## Prices

Prices are in US dollars per one million tokens. `M` means one million.

| Provider | Input price | Output price | Price basis |
| --- | ---: | ---: | --- |
| `gpt-5.6-luna` | $0.20/M | $1.20/M | The `[cost]` section in `config.toml`. |
| DeepSeek Flash | $0.15/M | $0.60/M | DeepSeek V4.1 Flash off-peak, cache miss. Peak rates are $0.30/M and $1.20/M. |
| Jev from TypeSafe AI | $0.042/M | Free | Jev bills input tokens only. |

Sources: [DeepSeek model pricing](https://api-docs.deepseek.com/quick_start/pricing/)
and [TypeSafe AI's Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

For every row:

```text
Luna price = 0.20 × Luna input tokens / M + 1.20 × Luna output tokens / M
DeepSeek price = 0.15 × DeepSeek input tokens / M + 0.60 × DeepSeek output tokens / M
Jev price = 0.042 × Jev input tokens / M
```

## How to read the tables

- `Luna tokens` and `DeepSeek tokens` use `input + output = total`.
- `Jev tokens` lists input tokens only because Jev output is free.
- Variables such as `Lstd_in` and `Dstd_out` are record-dependent. The adapter
  must supply their values for the record under study.
- The Neo4j and Milvus rows have the same model-token estimate. Milvus adds
  vector-storage and embedding costs, which are outside these LLM columns.
- The `vector DB only` and `Neo4j + Milvus` rows describe the planned split-store
  comparison. The current GraphRAG path stores graph data and vectors in Neo4j.
- Jev is an evaluation alternative in this table. The current DeepEval code uses
  `gpt-5.6-luna`; it does not call Jev.

The Streamlit run wraps each model-using stage with LangChain's native
`get_usage_metadata_callback()`. It appends the returned `usage_metadata` to
`runs/usage-*.jsonl` and displays the same values.
The current `neo4j-graphrag` search call combines retrieval and answer
generation, so those calls are recorded together. Splitting them would require
reimplementing that package flow.

## HotpotQA

| Stage | Luna tokens | DeepSeek tokens | Jev tokens | Luna price | DeepSeek price | Jev price |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1a. Standard construction, Neo4j only | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1a. Standard construction, Neo4j and Milvus | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1b. Ontology-guided construction, Neo4j only | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1b. Ontology-guided construction, Neo4j and Milvus | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1c. Construction evaluation | `Lce_in + Lce_out` | `Dce_in + Dce_out` | `Jce_in` | `Lce_cost` | `Dce_cost` | `0.042 × Jce_in / M` |
| 2a. Agentic retrieval | `Lar_in + Lar_out` | `Dar_in + Dar_out` | — | `Lar_cost` | `Dar_cost` | — |
| 2b. Vector DB only, no graph | `Lv_in + Lv_out` | `Dv_in + Dv_out` | — | `Lv_cost` | `Dv_cost` | — |
| 2c. Vector plus Cypher and graph | `Lvg_in + Lvg_out` | `Dvg_in + Dvg_out` | — | `Lvg_cost` | `Dvg_cost` | — |
| 2d. Retrieval evaluation | `Lre_in + Lre_out` | `Dre_in + Dre_out` | `Jre_in` | `Lre_cost` | `Dre_cost` | `0.042 × Jre_in / M` |
| 3. Answer generation | `La_in + La_out` | `Da_in + Da_out` | — | `La_cost` | `Da_cost` | — |

## 2WikiMultiHopQA

| Stage | Luna tokens | DeepSeek tokens | Jev tokens | Luna price | DeepSeek price | Jev price |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1a. Standard construction, Neo4j only | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1a. Standard construction, Neo4j and Milvus | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1b. Ontology-guided construction, Neo4j only | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1b. Ontology-guided construction, Neo4j and Milvus | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1c. Construction evaluation | `Lce_in + Lce_out` | `Dce_in + Dce_out` | `Jce_in` | `Lce_cost` | `Dce_cost` | `0.042 × Jce_in / M` |
| 2a. Agentic retrieval | `Lar_in + Lar_out` | `Dar_in + Dar_out` | — | `Lar_cost` | `Dar_cost` | — |
| 2b. Vector DB only, no graph | `Lv_in + Lv_out` | `Dv_in + Dv_out` | — | `Lv_cost` | `Dv_cost` | — |
| 2c. Vector plus Cypher and graph | `Lvg_in + Lvg_out` | `Dvg_in + Dvg_out` | — | `Lvg_cost` | `Dvg_cost` | — |
| 2d. Retrieval evaluation | `Lre_in + Lre_out` | `Dre_in + Dre_out` | `Jre_in` | `Lre_cost` | `Dre_cost` | `0.042 × Jre_in / M` |
| 3. Answer generation | `La_in + La_out` | `Da_in + Da_out` | — | `La_cost` | `Da_cost` | — |

## Natural Questions

| Stage | Luna tokens | DeepSeek tokens | Jev tokens | Luna price | DeepSeek price | Jev price |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 1a. Standard construction, Neo4j only | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1a. Standard construction, Neo4j and Milvus | `Lstd_in + Lstd_out` | `Dstd_in + Dstd_out` | — | `Lstd_cost` | `Dstd_cost` | — |
| 1b. Ontology-guided construction, Neo4j only | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1b. Ontology-guided construction, Neo4j and Milvus | `Log_in + Log_out` | `Dog_in + Dog_out` | — | `Log_cost` | `Dog_cost` | — |
| 1c. Construction evaluation | `Lce_in + Lce_out` | `Dce_in + Dce_out` | `Jce_in` | `Lce_cost` | `Dce_cost` | `0.042 × Jce_in / M` |
| 2a. Agentic retrieval | `Lar_in + Lar_out` | `Dar_in + Dar_out` | — | `Lar_cost` | `Dar_cost` | — |
| 2b. Vector DB only, no graph | `Lv_in + Lv_out` | `Dv_in + Dv_out` | — | `Lv_cost` | `Dv_cost` | — |
| 2c. Vector plus Cypher and graph | `Lvg_in + Lvg_out` | `Dvg_in + Dvg_out` | — | `Lvg_cost` | `Dvg_cost` | — |
| 2d. Retrieval evaluation | `Lre_in + Lre_out` | `Dre_in + Dre_out` | `Jre_in` | `Lre_cost` | `Dre_cost` | `0.042 × Jre_in / M` |
| 3. Answer generation | `La_in + La_out` | `Da_in + Da_out` | — | `La_cost` | `Da_cost` | — |

## Variable names

The price cells use the same suffix as the token cells. For example,
`Lstd_cost` is `0.20 × Lstd_in / M + 1.20 × Lstd_out / M`.

| Variable | Meaning |
| --- | --- |
| `std` | Standard graph extraction for one record. |
| `og` | Ontology-guided graph extraction for one record. |
| `ce` | Construction evaluation for one graph. |
| `ar` | Agentic retrieval and its tool decisions. |
| `v` | Vector-only retrieval. |
| `vg` | Vector retrieval with Cypher and graph context. |
| `re` | Retrieval evaluation for one retrieved result. |
| `a` | Final answer generation for one question. |
| `L`, `D`, `J` | Luna, DeepSeek, and Jev. The suffix `_in` means input tokens and `_out` means output tokens. |

The answer output is small for these datasets. Record it for completeness, but
expect construction and evaluation to dominate the total.
