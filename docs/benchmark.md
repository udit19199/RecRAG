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
Agentic Hybrid RAG comparison. The official-source loaders remain the default.
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

## Differences from the paper

The paper's HotpotQA results use an unpublished random sample of 1,000 hard
bridge questions. RecRAG's local
`datasets/hotpotqa/hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json` uses a
reproducible sample from the official validation split. Its exact questions
still differ from the paper's unpublished sample. Run with `--records 1000` to
answer the full sample. Each query searches unique context pages across those
1,000 rows. The run's `corpus.json` records the question range, page count,
chunk count, and corpus hash.

HotpotQA answer precision, recall, and F1 use the official evaluator's
normalization, token overlap, and yes/no handling. The paper's evaluation code
calls `calculate_metrics`, but that helper is not included in its linked repo,
so we cannot verify exact formula parity. The paper reports per-question
precision, recall, and F1 averaged across questions. The paper's protocol saves
retrieved evidence, then applies one generation script with Llama 3.1 8B or
70B. RecRAG uses `gpt-6-luna`, and its Agentic Hybrid RAG is not the
paper's Microsoft Community-GraphRAG. Our score formula follows HotpotQA's
official scorer, but exact parity with the paper's helper is unverified.

For paper comparisons, report HotpotQA precision, recall, and F1, and
MultiHop-RAG `overlap_accuracy`. Exact match is an additional diagnostic,
not the paper's headline metric. Both QA scorers extract the authors' answer
wrapper `The answer to the question is "..."` before scoring. MultiHop-RAG
matches the published evaluator: trim the prediction, remove its periods,
lowercase prediction and reference, and accept any shared whitespace-delimited
word. It does not require exact yes/no answers or use an LLM judge.

The upstream HippoRAG source stays unchanged. The paper used
`text-embedding-ada-002`,
GPT-4o-mini for HippoRAG construction, and Llama-3.1-8B-Instruct for answers.
These runs instead use the shared RecRAG models above, so the paper's scores
cannot be compared directly with them. The paper also does not pin its
HippoRAG commit or publish its processed input records.

# Dataset choices and source records

RecRAG currently supports HotpotQA, MultiHop-RAG, Natural Questions, and
NovelQA. A **record** is one dataset item. It contains one question and the text
supplied with that item. Public NovelQA files do not include gold answers, so
its full-set runs have no answer scores. The publisher provides a separate
labeled demonstration and requires a request for full labels.

The datasets store that text in different shapes:

| Dataset | Text stored in each record |
| --- | --- |
| HotpotQA | Sentences grouped under page titles |
| MultiHop-RAG | News documents from a shared 609-document corpus |
| Natural Questions | The full Wikipedia article attached to the question |
| NovelQA | The public-domain novel attached to its questions |

Compare results within each dataset. The sources use different document
collection sizes and answer formats.

The example below shows one record in a readable form.

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
