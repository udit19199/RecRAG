Subject: RecRAG held-out 400: retrieval by question type, across 3 runs

Hi all,

Following my 3 Nov mail: retrieval broken down by question type, 100 questions per type, 3 runs. Both retrieval checks are now complete for all runs.

Questions that pass in all 3 runs (out of 100):

| Type | Method | Gold sentences ≥50% | Gold sentences 100% | Answer supported | Answered right |
|---|---|---|---|---|---|
| Comparison | Agentic GraphRAG | 62 | 9 | 24 | 35 |
|  | RAG | 68 | 19 | 35 | 46 |
|  | HippoRAG | 63 | 23 | 40 | 44 |
|  | RAPTOR | 66 | 21 | 34 | 43 |
| Temporal | Agentic GraphRAG | 48 | 13 | 27 | 43 |
|  | RAG | 52 | 15 | 36 | 43 |
|  | HippoRAG | 51 | 14 | 40 | 56 |
|  | RAPTOR | 48 | 16 | 33 | 44 |
| Inference | Agentic GraphRAG | 34 | 6 | 17 | 95 |
|  | RAG | 44 | 11 | 22 | 94 |
|  | HippoRAG | 33 | 8 | 24 | 97 |
|  | RAPTOR | 44 | 12 | 22 | 98 |

- The baselines give the same retrieval result in every run. Agentic GraphRAG does not: its result changes between runs on 16 comparison, 27 temporal, 20 inference questions (gold sentences ≥50%) and 26 comparison, 26 temporal, 26 inference (answer support).
- In any single run Agentic GraphRAG is close to the baselines; it falls behind on questions it gets right in every run.
- Answer support is judged by gpt-6-luna: does the retrieved text state what is needed to reach the answer, by any route.

Full per-run confusion matrices (Agentic GraphRAG vs each baseline, every threshold) are on slides 11 and 12 of the Experiments scene.

Warm Regards,
