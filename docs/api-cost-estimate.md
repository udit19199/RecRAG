# API cost estimate

This page lists the four dataset adapters and how the app measures API cost.
The active app scope and method names are in the [documentation index](index.md).

## Datasets

The app can run the same pipeline on these inputs:

| Dataset | Source | Adapter |
| --- | --- | --- |
| [HotpotQA](https://hotpotqa.github.io/) | `datasets/hotpotqa/distractor-validation.parquet` | `hotpotqa` |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | `datasets/triviaqa/rc.wikipedia-validation.parquet` | `triviaqa` |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | `datasets/2wikimultihopqa/dev.json` | `2wikimultihopqa` |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions) | A dev JSONL file under `datasets/natural_questions/` | `natural_questions` |

The first three inputs are present in this workspace. Natural Questions is
optional because its raw JSONL file is not checked into the repository.

Each adapter maps its source record to the same pipeline fields:

| Field | Use |
| --- | --- |
| Question | Retrieval and answer generation. |
| Source pages and passages | Graph construction. |
| Answer and aliases | Answer evaluation. |
| Supporting passages | Retrieval and answer evaluation. |

## Cost measurement

The Streamlit run reports model calls, input tokens, output tokens, and an
estimated USD cost after processing the selected records. One shared ledger
counts calls from graph construction, agentic or vector retrieval, answer
generation, and DeepEval judges.

The default rates in `config.toml` are:

| Model | Input / 1M tokens | Output / 1M tokens |
| --- | ---: | ---: |
| GPT-5.6 Luna | $0.20 | $1.20 |

The estimate is:

```text
input_tokens / 1,000,000 * input_rate
+ output_tokens / 1,000,000 * output_rate
```

Run the same record count, construction methods, retrieval methods, and
evaluation setting for each dataset when comparing cost. The ledger excludes
embedding tokens, Neo4j, machine, storage, network, and retry costs.
