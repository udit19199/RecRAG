# HotpotQA 100: evidence scores by question type (rebuilt 2026-11-08)

The original HotpotQA 100 run folders (12 to 13 Oct 2026) were deleted. Their
overall results are on slide 9 of the Experiments Excalidraw scene, which is the
source of truth. This folder only holds the per-type split added on slides 5 to 7.

- `rebuild_scores.py`: rebuilds answer-support and gold-sentence scores per
  question and method from the gpt-6-luna verdicts kept in
  `runs/retrieval_judge/cache.jsonl`. The retrieved text is gone, so quotes are
  not checked against it; totals come out about 1 to 2 points above the originals.
- `split_by_type.py`: labels the 100 questions with MultiHop-RAG's four types and
  writes `scores_by_type.json`. Labels (question index in the hard-bridge sample):
  temporal = 2, 6, 17, 31, 51, 75, 78, 85 (asks for a date or "before");
  comparison = 18, 25, 54 (a shared attribute of named entities);
  null = none; inference = the other 89.
- Accuracy cannot be split by type: the per-question answers were deleted.

Run from the repo root: `python3 runs/hotpotqa100/rebuild_scores.py && python3 runs/hotpotqa100/split_by_type.py`.
