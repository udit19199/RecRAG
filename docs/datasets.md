# Selected datasets

## Scope

We reviewed the 400 papers in [paper-index.md](../.context/paper-index.md) and
kept the three datasets with the best fit for this benchmark:

| Dataset | Role | Why it is useful here |
| --- | --- | --- |
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA | Questions require evidence from more than one Wikipedia page. |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | Multi-hop QA | Includes supporting facts and explicit evidence paths. |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | Open-domain QA | Includes question-answer pairs and the Wikipedia evidence paragraphs used to answer them. |

MuSiQue is not included. The three datasets above are the working set for the
benchmark.

## 500 versus 1,000 examples

The counts below are paper counts from the 400-paper index. We counted a paper
only when the full text stated that it evaluated a fixed set of 500 or 1,000
questions, examples, or queries from that dataset. We did not count token,
passage, chunk, top-k, or training-step counts. The same paper can appear in
both columns when it used different sizes for different datasets or runs.

| Dataset | Papers using 500 | Papers using 1,000 |
| --- | ---: | ---: |
| HotpotQA | 28 | 38 |
| 2WikiMultiHopQA | 18 | 22 |
| TriviaQA | 6 | 2 |
| **Total dataset-paper uses** | **52** | **62** |

For these three datasets, 1,000 examples is more common, but the difference is
not large. **500 examples is a safe default** and leaves 1,000 as an optional
larger run.

Papers that support the 500-example choice include:

- [π-CoT](https://arxiv.org/abs/2506.20642): 500 questions from each multi-hop
  dataset.
- [ReARTeR](https://arxiv.org/abs/2501.07861): 500 development examples from
  HotpotQA and 2WikiMultiHopQA.
- [EcphoryRAG](https://arxiv.org/abs/2510.08958): a random sample of 500
  questions per dataset.
- [PersonalAI 2.0](https://arxiv.org/abs/2605.13481): 500 TriviaQA pairs and
  1,000 pairs from each of its multi-hop datasets.

## Local dataset files

The `datasets/` directory contains benchmark-ready evaluation data and the
official source repositories. The downloaded files are local-only because the
repository ignores dataset downloads.

| Dataset | Local data | Records |
| --- | --- | ---: |
| HotpotQA | `distractor-validation.parquet`, `fullwiki-validation.parquet` | 7,405 each |
| 2WikiMultiHopQA | `dev.json`, `id_aliases.json` | 12,576 questions |
| TriviaQA | `rc.wikipedia-validation.parquet` | 7,993 questions |

The TriviaQA file is the `rc.wikipedia` split, not a metadata-only sample. Each
row contains `question`, `answer`, and Wikipedia evidence in
`entity_pages.wiki_context`.

## Example records

These are exact question-answer values from the local files:

| Dataset | Question | Answer |
| --- | --- | --- |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes |
| 2WikiMultiHopQA | Who is the mother of the director of film Polish-Russian War (Film)? | Małgorzata Braunek |
| TriviaQA | Which Lloyd Webber musical premiered in the US on 10th December 1993? | Sunset Boulevard |

The corresponding TriviaQA row also has `question_id: tc_33`, the Wikipedia
page title `Andrew Lloyd Webber`, and the full evidence paragraph in
`entity_pages.wiki_context`.

## Source repositories

- [HotpotQA source](https://github.com/hotpotqa/hotpot)
- [2WikiMultiHopQA source](https://github.com/Alab-NII/2wikimultihop)
- [TriviaQA source](https://github.com/mandarjoshi90/triviaqa)
