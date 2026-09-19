# API cost estimate

## Pricing

| Model | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| GPT-5.6 Luna | $0.20 | $1.20 |
| DeepSeek V4.1 Flash on Fireworks | $0.22 | $0.66 |

Prices:
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [DeepSeek on Fireworks](https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash).

## Tokens and cost per record

The estimate uses the current average of 4.66 chunks per record. Costs below are planning estimates, not provider usage logs.

The rows below are the per-graph and per-result inputs for one earlier run.
That run combined both construction methods and the benchmark retrieval methods.

| Process | Luna input / output / total | DeepSeek input / output / total | Luna cost | DeepSeek cost |
| --- | --- | --- | ---: | ---: |
| Standard construction | 9.99k / 8.49k / **18.48k** | 9.99k / 12.74k / **22.73k** | $0.0122 | $0.0106 |
| Ontology-guided construction | 7.68k / 6.99k / **14.67k** | 7.68k / 10.49k / **18.17k** | $0.0099 | $0.0086 |
| Agentic retrieval, per graph | 7.80k / 1.41k / **9.21k** | 7.80k / 2.12k / **9.92k** | $0.0033 | $0.0031 |
| Vector retrieval, per graph | 1.80k / 0.36k / **2.16k** | 1.80k / 0.54k / **2.34k** | $0.0008 | $0.0008 |
| Construction evaluation, per graph | 6.10k / 1.80k / **7.90k** | 6.10k / 2.70k / **8.80k** | $0.0034 | $0.0031 |
| Retrieval evaluation, per result | 10.40k / 5.25k / **15.65k** | 10.40k / 7.88k / **18.28k** | $0.0084 | $0.0075 |
| Answer evaluation, per result | 5.80k / 4.44k / **10.24k** | 5.80k / 6.66k / **12.46k** | $0.0065 | $0.0057 |

Evaluation costs more because it runs for every graph or result and sends the graph, source passages, retrieved context, or answer to the judge.
A metric can also use several judge requests to return one score and reason.

## Earlier estimate

This describes an earlier run with both construction methods, agentic retrieval, vector retrieval, answer generation, and all evaluations.

| Records | Luna tokens total | DeepSeek tokens total | Luna cost | DeepSeek cost |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 118.13M | 138.72M | **$64.82** | **$57.70** |
| 1,000 | 236.25M | 277.44M | **$129.63** | **$115.41** |

The rows above are projections for the two requested record counts.

Per record:

| Model | Input | Output and reasoning | Total |
| --- | ---: | ---: | ---: |
| Luna | 153.87k | 82.38k | **236.25k** |
| DeepSeek | 153.87k | 123.57k | **277.44k** |

## Cost reduction plan

Each alternative starts from the earlier estimate. Do not combine rows. Savings are shown as Luna input/output tokens saved per record.
DeepSeek output savings are 1.5x higher.

The estimate covers agentic and vector retrieval.

| Plan | Saved per record, Luna input / output | Luna total tokens / record | DeepSeek total tokens / record | 500 Luna / DeepSeek | 1,000 Luna / DeepSeek |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier three-method setup | — | 236.25k | 277.44k | $64.82 / $57.70 | $129.63 / $115.41 |
| LLM evaluation on 10% sample | 127.62k / 73.01k | 91.72k | 106.46k | $23.91 / $21.44 | $47.82 / $42.87 |
