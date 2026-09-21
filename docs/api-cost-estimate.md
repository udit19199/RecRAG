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

## TriviaQA

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
