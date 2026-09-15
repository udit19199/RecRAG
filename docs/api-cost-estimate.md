# Cost and model estimates

This page explains what the full 2WikiMultiHopQA comparison costs, why the
cost grows, and which assumptions produce each number.

These are planning estimates. They are not billing records from the app. The
prices below were checked on 2026-09-16. All prices are in US dollars and use
standard, uncached API rates.

## Short answer

One full record runs:

```text
2 graph builds
8 retrieval and answer paths
2 graph evaluations
8 retrieval evaluations
8 answer evaluations
```

With the current token assumptions, one record costs about:

| Model scenario | LLM tokens | LLM cost |
| --- | ---: | ---: |
| `gpt-5.6-luna`, current model | 292.35k | **$0.16095** |
| DeepSeek V4.1 Flash, planning scenario | 343.59k | **$0.14323** |

The Luna estimate is about **$804.75 for 5,000 records**. The DeepSeek number
is lower because the planning scenario uses a lower output-token price. It is
not a result from the current app.

The totals above cover LLM input and output tokens only. Embedding requests,
Neo4j, compute, network, retries, failed requests, and reruns are separate.

## What one record means

One record is one item from `datasets/2wikimultihopqa/dev.json`. The full run
uses both construction methods and all four retrieval methods:

| Choice | Values | Count per record |
| --- | --- | ---: |
| Construction | `standard`, `ontology_guided` | 2 graph builds |
| Retrieval | `text2cypher`, `agentic`, `vector`, `hybrid` | 8 paths, 4 per graph |
| Graph evaluation | groundedness, completeness, supporting-evidence coverage | 2 graph evaluations |
| Retrieval evaluation | contextual precision, recall, relevancy | 8 retrieval evaluations |
| Answer evaluation | faithfulness, relevancy, correctness, alias match | 8 answer evaluations |

The 2Wiki `dev.json` file has 12,576 records. Across that file, one record has
an average of 10 source pages, 31.88 passages, and 4.6555 text chunks. The
construction code uses 1,000-character chunks with 100 characters of overlap.
The estimate rounds 4.6555 chunks to 4.66.

## What the app calls

The app uses the same `gpt-5.6-luna` model for extraction, retrieval decisions,
answer generation, and evaluation judges. It uses `text-embedding-3-small`
for vector search. The LLM uses medium reasoning effort.

The work has different request types. A graph build, an LLM request, an
embedding request, and an evaluation metric are not the same unit.

| Stage | What happens for one record | LLM requests | Embedding requests |
| --- | --- | ---: | ---: |
| Graph construction | Build one graph from about 4.66 chunks. Run this for two methods. | About 4.66 extraction requests per graph | About 4.66 chunk embeddings per graph |
| `text2cypher` path | Generate Cypher, run it, then generate an answer. Run this on both graphs. | 2 per path | 0 |
| `vector` path | Embed the question, search the vector index, then generate an answer. | 1 per path | 1 per path |
| `hybrid` path | Embed the question, run vector plus full-text search, then generate an answer. | 1 per path | 1 per path |
| `agentic` path | Let an agent choose tools, then generate an answer. | Up to 5 agent calls, plus one Cypher-generation call per `cypher_search`, plus one answer call | One per `vector_search` |
| Graph evaluation | Score groundedness, completeness, and supporting-evidence coverage. | 3 judge calls per graph | 0 |
| Retrieval evaluation | Score contextual precision, recall, and relevancy. | 6 judge calls per result | 0 |
| Answer evaluation | Check the alias match, faithfulness, relevancy, and correctness. | Up to 8 judge calls per result | 0 |

The agent has a five-call limit in `graphrag/retrieval/agentic.py`. A call to
`cypher_search` also calls the text-to-Cypher LLM. The exact agent cost depends
on which tools it chooses and how many times it searches.

The answer evaluation uses a deterministic alias check. That check makes no
API request. When retrieved context exists, the other three answer metrics use
the following judge calls:

| Metric | Judge calls | Why |
| --- | ---: | --- |
| Faithfulness | 4 | Extract truths, extract claims, compare claims, write a reason |
| Answer relevancy | 3 | Extract statements, judge them, write a reason |
| Answer correctness | 1 | Judge the answer against the reference answer and aliases |

If retrieval returns no context, the app skips the faithfulness metric. The
full estimate assumes that context exists.

## Pricing used in the estimate

The estimate uses the standard token rates below. It does not use cached-input,
Batch API, or Fast mode rates.

| Model | Provider | Input / 1M tokens | Cached input / 1M tokens | Output / 1M tokens | Role |
| --- | --- | ---: | ---: | ---: | --- |
| `gpt-5.6-luna` | OpenAI | $0.20 | $0.02 | $1.20 | Current model |
| DeepSeek V4.1 Flash | Fireworks | $0.22 | $0.007 | $0.66 | Planning scenario only |

Sources: [GPT-5.6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[DeepSeek V4.1 Flash pricing](https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash),
and [OpenAI pricing](https://developers.openai.com/api/docs/pricing).

The current code configures `ChatOpenAI` with `gpt-5.6-luna`. The DeepSeek
scenario would need a provider adapter or endpoint configuration before it
could run. It must not be read as a completed model comparison.

The current embedding model costs $0.02 per 1M input tokens. Embeddings have no
output-token charge. See [text-embedding-3-small pricing](https://developers.openai.com/api/docs/models/text-embedding-3-small).

## Token assumptions for one unit

The table below is the complete input and output budget used by the estimate.
The input includes the prompt, schema, question, source text, graph output, or
retrieved context sent to that unit. The output includes the model response and
reasoning tokens when the provider bills them as output.

| Unit | What the unit includes | Luna input | Luna output | DeepSeek input | DeepSeek output |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard graph build, per graph | One extraction request per average chunk | 9.99k | 8.49k | 9.99k | 12.74k |
| Ontology-guided graph build, per graph | One extraction request per average chunk with the guided schema | 7.68k | 6.99k | 7.68k | 10.49k |
| `text2cypher` path, per graph | Cypher generation plus answer generation | 3.80k | 0.81k | 3.80k | 1.22k |
| `agentic` path, per graph | Agent decisions, tool calls, and answer generation | 7.80k | 1.41k | 7.80k | 2.12k |
| `vector` path, per graph | Answer generation; question embedding is separate | 1.80k | 0.36k | 1.80k | 0.54k |
| `hybrid` path, per graph | Answer generation; question embedding is separate | 1.80k | 0.36k | 1.80k | 0.54k |
| Graph evaluation, per graph | Three graph judge metrics | 6.10k | 1.80k | 6.10k | 2.70k |
| Retrieval evaluation, per result | Three contextual judge metrics | 10.40k | 5.25k | 10.40k | 7.88k |
| Answer evaluation, per result | Three LLM judge metrics; alias matching is deterministic | 5.80k | 4.44k | 5.80k | 6.66k |

The DeepSeek input values are the same as Luna. The DeepSeek output values are
1.5 times the Luna output values. That 1.5 multiplier is a planning assumption,
not a measured result.

These row values are estimated prompt and output sizes. They are inputs to the
calculation, not usage values emitted by the current app. The `agentic` row is
an average planning allowance. Its five-call limit is a request limit, not a
token limit.

## The full calculation

The full run uses each construction method once, each retrieval method on both
graphs, and each evaluation for every graph or result.

### Luna input tokens

```text
9.99 + 7.68
+ 2 x (3.80 + 7.80 + 1.80 + 1.80)
+ 2 x 6.10
+ 8 x 10.40
+ 8 x 5.80
= 189.87k input tokens
```

### Luna output tokens

```text
8.49 + 6.99
+ 2 x (0.81 + 1.41 + 0.36 + 0.36)
+ 2 x 1.80
+ 8 x 5.25
+ 8 x 4.44
= 102.48k output tokens
```

### DeepSeek output tokens

```text
1.5 x 102.48k
= 153.72k output tokens
```

This uses the unrounded Luna total. Multiplying the rounded values in the unit
table can produce a small rounding difference.

The cost formula is:

```text
cost = input_tokens / 1,000,000 x input_price
     + output_tokens / 1,000,000 x output_price
```

For Luna:

```text
(189,870 / 1,000,000 x $0.20)
+ (102,480 / 1,000,000 x $1.20)
= $0.16095 per record
```

For the DeepSeek planning scenario:

```text
(189,870 / 1,000,000 x $0.22)
+ (153,720 / 1,000,000 x $0.66)
= $0.14323 per record
```

This is the cost of a full matrix for one record. It is not the cost of one
single question answered with one retrieval method.

## Where the cost goes

| Stage | Luna input | Luna output | Luna cost per record | Share of Luna cost |
| --- | ---: | ---: | ---: | ---: |
| Graph construction | 17.67k | 15.48k | $0.02211 | 14% |
| Retrieval and answer generation | 30.40k | 5.88k | $0.01314 | 8% |
| All evaluations | 141.80k | 81.12k | $0.12570 | 78% |
| **Total** | **189.87k** | **102.48k** | **$0.16095** | **100%** |

Evaluation is the bottleneck because the app judges every graph and every
retrieval result after generation. Evaluation uses 222.92k of the 292.35k
estimated LLM tokens per record. That is 76% of the total token volume and 78%
of the Luna cost.

Agentic retrieval also costs more than vector or hybrid retrieval because it can
make several model calls before the answer is generated. Its evaluation costs
the same as every other retrieval result.

## Cost at different record counts

The total scales linearly when the average record stays the same.

| Records | Luna | DeepSeek planning scenario |
---:|---:|---:|
| 1 | $0.16 | $0.14 |
| 100 | $16.10 | $14.32 |
| 1,000 | $160.95 | $143.23 |
| 5,000 | **$804.75** | **$716.13** |
| 12,576, full `dev.json` | **$2,024.11** | **$1,801.22** |

These totals are rounded to cents. They assume every record has the dataset
average of 4.66 chunks and the same output lengths as the unit table.

## What changes the cost

Each row below starts from the full Luna setup. The rows are independent. Do
not add them together.

| Change | Input tokens / record | Output tokens / record | Luna cost for 5,000 records | What it removes |
| --- | ---: | ---: | ---: | --- |
| Full setup | 189.87k | 102.48k | **$804.75** | Nothing |
| Remove `agentic` | 141.87k | 80.28k | **$623.55** | Its two paths and their retrieval and answer evaluations |
| Remove graph evaluation | 177.67k | 98.88k | $770.95 | Two graph evaluations |
| Remove retrieval evaluation | 106.67k | 60.48k | $469.55 | Eight retrieval evaluations |
| Remove LLM answer evaluation | 143.47k | 66.96k | $545.23 | Eight answer evaluations; alias matching remains possible |
| Run no LLM evaluations | 48.07k | 21.36k | **$176.23** | All graph, retrieval, and answer judge calls |
| Run LLM evaluations on 10% of records | 62.25k | 29.47k | $239.08 | Evaluate one record in ten while still generating every answer |

The largest saving comes from reducing evaluation calls. Removing a retrieval
method also removes its answer and its evaluations, which is why the saving is
larger than the retrieval row alone.

The current Streamlit app enables all evaluations with the **Score with
DeepEval** checkbox. The app does not currently expose separate switches for
the reduction rows above. They are planning scenarios, not current UI options.

## Comparing model choices

The table below applies the current Luna token volume to several model prices.
It compares price only. It does not predict answer quality, output length, tool
choices, or reliability.

| Model | Input / 1M | Output / 1M | Assumed output tokens | Cost per record | Cost for 5,000 records | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gpt-5.6-luna` | $0.20 | $1.20 | 102.48k | $0.16095 | $804.75 | Current |
| `gpt-5.6-terra` | $2.00 | $12.00 | 102.48k | $1.60950 | $8,047.50 | Price-only scenario |
| `gpt-5.6-sol` | $4.00 | $20.00 | 102.48k | $2.80908 | $14,045.40 | Price-only scenario |
| DeepSeek V4.1 Flash | $0.22 | $0.66 | 153.72k | $0.14323 | $716.13 | Provider scenario |

For a real model comparison, keep the following fixed:

- the records and dataset split;
- the chunk size and overlap;
- the construction and retrieval methods;
- the prompts and schemas;
- the answer settings;
- the evaluation definitions.

Change only the model. Record the graph scores, retrieval scores, answer
scores, input tokens, output tokens, model calls, and cost. A cheaper model can
still cost more if it produces longer outputs, takes more agent steps, or causes
more reruns.

The current code uses only `gpt-5.6-luna`. The other rows do not mean that those
models have been tested in this repository.

## Embedding cost

The LLM totals do not include embeddings. The current code calls
`text-embedding-3-small` in these places:

- The graph pipeline embeds every text chunk once. Two graph builds use about
  `2 x 4.66 = 9.31` chunk embedding requests per record.
- Each `vector` path embeds the question once. There are four such paths per
  record because two graphs use the vector method.
- Each `hybrid` path also embeds the question once. There are four such paths.
- Each `vector_search` tool call inside an `agentic` path embeds the question.
  The number depends on the agent's tool choices.
- `text2cypher` does not need an embedding.

Embedding cost is:

```text
embedding_cost = embedding_input_tokens / 1,000,000 x $0.02
```

The current app does not save embedding token usage, so this page does not add
a guessed embedding amount to the LLM total. The exact amount must come from
the embedding provider's reported input-token usage.

## What the estimate excludes

The displayed totals exclude:

- embedding input tokens;
- Neo4j Desktop or server cost;
- CPU, memory, disk, and network cost;
- taxes, credits, currency conversion, and account fees;
- cached-input or Batch API discounts;
- retries, failed requests, and reruns;
- extra agent calls caused by future prompt or middleware changes.

The full 292.35k Luna tokens are spread across many requests. They do not by
themselves trigger long-context pricing. A long-context price depends on the
input size of one request, not the sum of all requests for a record.

The app does not currently persist provider usage for each stage. Until that
instrumentation exists, the unit table remains an estimate. An actual cost
report must record `prompt_tokens`, `output_tokens`, cached tokens when present,
the model, and the stage for every response.

## Source of each assumption

The implementation that produces these counts is here:

- [`streamlit_app.py`](../streamlit_app.py) loops over records, construction methods, and retrieval methods.
- [`construction.py`](../graphrag/construction/construction.py) sets the chunk size, overlap, and graph pipeline.
- [`answering.py`](../graphrag/retrieval/answering.py) runs one retrieval and answer path for each method.
- [`agentic.py`](../graphrag/retrieval/agentic.py) sets the five-call agent limit and tool choices.
- [`evals/construction.py`](../graphrag/evals/construction.py) defines graph and answer evaluation calls.
- [`evals/retrieval.py`](../graphrag/evals/retrieval.py) defines retrieval evaluation calls.
- [`two_wiki_multihopqa.py`](../graphrag/dataset_records/two_wiki_multihopqa.py) loads the records and supporting passages.
- [`config.toml`](../config.toml) sets the current LLM and embedding models.
