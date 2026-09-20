# Benchmark plan and cost

This page defines the benchmark data and the planning cost for a run. It keeps
three states separate:

- the 2WikiMultiHopQA run that the app supports today;
- the HotpotQA and TriviaQA benchmark plan;
- the older cost estimate used to compare model work.

## What runs today

The Streamlit app loads 2WikiMultiHopQA records through
[`load_records()`](../graphrag/dataset_records/two_wiki_multihopqa.py) and reads
`datasets/2wikimultihopqa/dev.json`. The UI accepts up to 20 records.

For each loaded record, the app can run:

- `standard` and `ontology_guided` construction;
- `agentic`, `vector`, and `entity_vector` retrieval;
- one answer for each construction and retrieval pair;
- optional construction, retrieval, and answer scoring with DeepEval.

The local HotpotQA and TriviaQA files are not connected to the app yet.

## Planned datasets

The first planned benchmark uses 500 questions from HotpotQA and 500 questions
from TriviaQA. 2WikiMultiHopQA remains the active dataset and an optional
500-question comparison dataset. The final choice of a third primary dataset is
still open.

| Dataset | Role | What it tests | Source |
| --- | --- | --- | --- |
| [HotpotQA](https://hotpotqa.github.io/) | Primary | Multi-hop answers that need more than one Wikipedia page. | `distractor-validation.parquet` or `fullwiki-validation.parquet` |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | Primary | Open-domain questions with Wikipedia evidence. | `rc.wikipedia-validation.parquet` |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Comparison | Multi-hop answers with supporting facts and explicit evidence paths. | `dev.json` or `dev.parquet` |

The local files contain these records:

| Dataset | Records | Fields used by the benchmark |
| --- | ---: | --- |
| HotpotQA, distractor split | 7,405 | `id`, `question`, `answer`, `supporting_facts`, `context` |
| HotpotQA, fullwiki split | 7,405 | `id`, `question`, `answer`, `supporting_facts`, `context` |
| 2WikiMultiHopQA | 12,576 | `question`, `context`, `supporting_facts`, `evidences`, `answer` |
| TriviaQA | 7,993 | `question`, `question_id`, `entity_pages`, `answer` |

Choose one HotpotQA validation file before the benchmark. Keep that choice
fixed across all methods. The two files contain the same questions with
different context orders and context sets.

TriviaQA stores Wikipedia evidence in `entity_pages.wiki_context`. Its answer
object also contains aliases for answer matching. HotpotQA stores supporting
page and sentence positions. 2WikiMultiHopQA stores `evidences` and
`evidences_id` for the explicit evidence-path comparison.

The loaders map each dataset to the same conceptual inputs:

| Input | Use |
| --- | --- |
| Question | Retrieval and answer generation. |
| Source page titles and passages | Graph construction. |
| Answer and aliases | Answer evaluation. |
| Supporting facts or evidence passages | Reference context for evaluation. |

The [construction reference](graphrag/construction.md) explains how source
pages become chunks, entities, relationships, and embeddings in Neo4j.

### Sample size

The planned run uses:

- 500 HotpotQA questions from the selected validation file;
- 500 TriviaQA questions from `rc.wikipedia-validation.parquet`;
- 500 2WikiMultiHopQA questions only if the comparison run is approved.

Save the split, sample-selection rule, and random seed in the run. A paper
review found 52 uses of 500 examples and 62 uses of 1,000 examples across the
three datasets. The first run uses 500 because it limits the cost while giving
each method the same question count.

### Local examples

| Dataset | Question | Answer |
| --- | --- | --- |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes |
| 2WikiMultiHopQA | Who is the mother of the director of film *Polish-Russian War (Film)*? | Małgorzata Braunek |
| TriviaQA | Which Lloyd Webber musical premiered in the US on 10th December 1993? | Sunset Boulevard |

## Cost estimate

These are planning estimates for one benchmark question. They are not provider
usage logs. The estimate covers the older 2 × 2 matrix: `standard` and
`ontology_guided` construction with `agentic` and `vector` retrieval. The
current app also supports `entity_vector`, which is not included in this table.

The app uses GPT-5.6 Luna. DeepSeek is a comparison model only.

| Model | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| GPT-5.6 Luna | $0.20 | $1.20 |
| DeepSeek V4.1 Flash on Fireworks | $0.22 | $0.66 |

[GPT-5.6 Luna pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[DeepSeek pricing](https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash).

### Cost by process

| Process | Calls / question | Luna input / output | Luna cost | DeepSeek input / output | DeepSeek cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard construction | 1 | 9.99k / 8.49k | $0.01219 | 9.99k / 12.74k | $0.01061 |
| Ontology-guided construction | 1 | 7.68k / 6.99k | $0.00992 | 7.68k / 10.49k | $0.00861 |
| Agentic retrieval, per graph | 2 | 7.80k / 1.41k | $0.00325 | 7.80k / 2.12k | $0.00312 |
| Vector retrieval, per graph | 2 | 1.80k / 0.36k | $0.00079 | 1.80k / 0.54k | $0.00075 |
| Answer generation, per result | 4 | 10.00k / 5.25k | $0.00830 | 10.00k / 7.875k | $0.00740 |
| Construction evaluation, per graph | 2 | 6.10k / 1.80k | $0.00338 | 6.10k / 2.70k | $0.00312 |
| Retrieval evaluation, per result | 4 | 10.40k / 5.25k | $0.00838 | 10.40k / 7.875k | $0.00749 |
| Answer evaluation, per result | 4 | 5.80k / 4.44k | $0.00649 | 5.80k / 6.66k | $0.00567 |

### Estimated run cost

| Run | Questions | Luna | DeepSeek comparison |
| --- | ---: | ---: | ---: |
| One question | 1 | $0.12963 | $0.11541 |
| HotpotQA | 500 | $64.82 | $57.70 |
| TriviaQA | 500 | $64.82 | $57.70 |
| Primary benchmark | 1,000 | **$129.63** | **$115.41** |
| 2WikiMultiHopQA, optional | 500 | $64.82 | $57.70 |

The primary benchmark estimate is **$129.63 with GPT-5.6 Luna** for 1,000
questions across HotpotQA and TriviaQA.

The estimate excludes embedding tokens, local Neo4j and machine costs,
retries, repeated runs, storage, and network costs.

## Dataset sources

- [HotpotQA source](https://github.com/hotpotqa/hotpot)
- [2WikiMultiHopQA source](https://github.com/Alab-NII/2wikimultihop)
- [TriviaQA source](https://github.com/mandarjoshi90/triviaqa)
