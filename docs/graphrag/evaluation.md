# GraphRAG evaluation

For each HotpotQA run, the Streamlit app builds one graph per selected
construction method from unique pages in the loaded validation rows, then
retrieves and answers each selected question from those graphs. A new run builds
new graphs. Other datasets still build graphs from each record's pages. Scores
are deterministic; the app does not call an LLM judge. NovelQA's full-set
inputs have no gold answers, so the app saves its generated answers without
answer scores.

## Answer scores

When gold answers are available, the app compares each generated answer with
the dataset answer and its aliases. NovelQA's full-set runs leave score fields
null. For HotpotQA, precision, recall, and F1 use the official evaluator's
normalization, token overlap, and yes/no handling. The paper's evaluation script
calls `calculate_metrics`, but that helper is not included in its linked repo,
so exact formula parity with the paper cannot be confirmed.

| Score | Meaning |
| --- | --- |
| `exact_match` | The normalized answer matches a gold answer exactly. |
| `precision`, `recall`, `f1` | Token overlap with the best-matching gold answer. |
| `overlap_accuracy` | The answer and canonical gold answer share at least one word. |
| `retrieval_accuracy` | The canonical answer text appears in the prompt built from retrieved context and the question. |

Exact match uses the same lowercase, punctuation, article, and whitespace
normalization. The paper reports mean per-question precision, recall, and F1.
The app also reports overlap accuracy and retrieval accuracy for continuity
with earlier runs.

The app also summarizes these scores by construction method, retrieval method,
and question type when the dataset provides one.

## Graph and run details

After construction, the app reports build time and graph counts: entities,
relationships, relationship types, duplicate entity names and nodes, isolated
entities, and self-loops. These counts describe graph shape, not graph quality.

For each answer, the app records retrieval-and-answer time, retrieved context,
and token usage. It saves per-record results under `runs/graphrag-*.jsonl`,
summary scores under `runs/graphrag-summary-*.json`, and model usage under
`runs/usage-*.jsonl`. These local run files are gitignored.

## Limits

- `retrieval_accuracy` checks for the answer string in the prompt. It is not a
  measure of supporting-passage recall.
- The datasets do not provide gold graph triples, so the app cannot calculate
  exact graph precision, recall, or F1.
- NovelQA's full-set inputs omit gold QA answers, so its answer scores are null.
- The app does not calculate summary scores because these runs have no reference
  summaries.

Older LLM-judge scores remain as historical results in the
[2WikiMultiHopQA report](2wikimultihopqa-results.md). The current app does not
regenerate them.
