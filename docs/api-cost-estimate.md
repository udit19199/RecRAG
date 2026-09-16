# API cost estimate

## Pricing

| Model | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| GPT-5.6 Luna | $0.20 | $1.20 |
| DeepSeek V4.1 Flash on Fireworks | $0.22 | $0.66 |

Prices: [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[DeepSeek on Fireworks](https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash).

## Tokens and cost per record

The estimate uses the current average of 4.66 chunks per record. Costs below are
planning estimates, not provider usage logs.

| Process | Luna input / output / total | DeepSeek input / output / total | Luna cost | DeepSeek cost |
| --- | --- | --- | ---: | ---: |
| Standard construction | 9.99k / 8.49k / **18.48k** | 9.99k / 12.74k / **22.73k** | $0.0122 | $0.0106 |
| Ontology-guided construction | 7.68k / 6.99k / **14.67k** | 7.68k / 10.49k / **18.17k** | $0.0099 | $0.0086 |
| Text-to-Cypher retrieval, per graph | 3.80k / 0.81k / **4.61k** | 3.80k / 1.22k / **5.02k** | $0.0017 | $0.0016 |
| Agentic retrieval, per graph | 7.80k / 1.41k / **9.21k** | 7.80k / 2.12k / **9.92k** | $0.0033 | $0.0031 |
| Vector retrieval, per graph | 1.80k / 0.36k / **2.16k** | 1.80k / 0.54k / **2.34k** | $0.0008 | $0.0008 |
| Hybrid retrieval, per graph | 1.80k / 0.36k / **2.16k** | 1.80k / 0.54k / **2.34k** | $0.0008 | $0.0008 |
| Construction evaluation, per graph | 6.10k / 1.80k / **7.90k** | 6.10k / 2.70k / **8.80k** | $0.0034 | $0.0031 |
| Retrieval evaluation, per result | 10.40k / 5.25k / **15.65k** | 10.40k / 7.88k / **18.28k** | $0.0084 | $0.0075 |
| Answer evaluation, per result | 5.80k / 4.44k / **10.24k** | 5.80k / 6.66k / **12.46k** | $0.0065 | $0.0057 |

Evaluation costs more because it runs for every graph or result and sends the
graph, source passages, retrieved context, or answer to the judge. A metric can
also use several judge requests to return one score and reason.

## Current full setup

This includes both construction methods, all four retrieval methods, answer
generation, and all evaluations.

| Records | Luna tokens total | DeepSeek tokens total | Luna cost | DeepSeek cost |
| ---: | ---: | ---: | ---: | ---: |
| 3,000 | 877.05M | 1,030.77M | **$482.85** | **$429.68** |
| 4,000 | 1,169.40M | 1,374.36M | **$643.80** | **$572.91** |
| 5,000 | 1,461.75M | 1,717.95M | **$804.75** | **$716.13** |

The rows above are projections for larger record counts.

Per record:

| Model | Input | Output and reasoning | Total |
| --- | ---: | ---: | ---: |
| Luna | 189.87k | 102.48k | **292.35k** |
| DeepSeek | 189.87k | 153.72k | **343.59k** |

## Cost reduction plan

Each alternative starts from the current full setup. Do not combine rows. Savings
are shown as Luna input/output tokens saved per record. DeepSeek output savings
are 1.5x higher.

The three-method scenario keeps agentic, vector, and hybrid retrieval. It drops
Text2Cypher because it was the weakest method in the result images.

| Plan | Saved per record, Luna input / output | Luna total tokens / record | DeepSeek total tokens / record | 4k Luna / DeepSeek | 5k Luna / DeepSeek |
| --- | --- | ---: | ---: | ---: | ---: |
| Current full setup | — | 292.35k | 343.59k | $643.80 / $572.91 | $804.75 / $716.13 |
| LLM evaluation on 10% sample | 127.62k / 73.01k | 91.72k | 106.46k | $191.27 / $171.49 | $239.08 / $214.36 |
| Two constructions and 3 retrieval methods only | 40.00k / 21.00k | **231.35k** | **272.09k** | **$511.00 / $454.55** | **$638.75 / $568.18** |
