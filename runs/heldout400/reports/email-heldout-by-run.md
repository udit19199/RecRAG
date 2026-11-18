Subject: RecRAG held-out 400: retrieval confusion matrices by question type, each run separately

Hi all,

Following my 3 Nov mail, here is the retrieval analysis broken down by question type, with each of the three runs shown separately (100 questions per type in every run), plus how consistent each question is across the runs. There are no overall averages here.

**How to read the tables**

- Retrieval check 1, gold-sentence recall: a question counts as *found* when at least 50%, 75% or 100% of its gold sentences are in the retrieved text.
- Retrieval check 2, answer support: a judge (gpt-6-luna) reads the retrieved text and scores 0–3 whether it states what is needed to reach the gold answer, by any valid route. *Answer stated* = score 2 or 3; *complete chain* = score 3. Now run on all 3 runs × 400 questions.
- Agentic GraphRAG vs baseline cells: **both / Agentic only / baseline only / neither**, out of 100 questions.
- Across-run cells: number of questions **in all 3 runs / 2 runs / 1 run / no run**.
- Null questions have no gold facts, so they appear only in the answer tables.

## Comparison

**Gold-sentence recall, Agentic GraphRAG vs each baseline, per run** (both / Agentic only / baseline only / neither)

|  | Run | Found ≥50% | Found ≥75% | Found 100% |
|---|---|---|---|---|
| vs RAG | Run 1 | 62 / 5 / 6 / 27 | 13 / 5 / 6 / 76 | 13 / 5 / 6 / 76 |
| vs RAG | Run 2 | 64 / 8 / 4 / 24 | 13 / 4 / 6 / 77 | 13 / 4 / 6 / 77 |
| vs RAG | Run 3 | 64 / 8 / 4 / 24 | 13 / 8 / 6 / 73 | 13 / 8 / 6 / 73 |
| vs HippoRAG | Run 1 | 57 / 10 / 6 / 27 | 13 / 5 / 10 / 72 | 13 / 5 / 10 / 72 |
| vs HippoRAG | Run 2 | 58 / 14 / 5 / 23 | 13 / 4 / 10 / 73 | 13 / 4 / 10 / 73 |
| vs HippoRAG | Run 3 | 59 / 13 / 4 / 24 | 15 / 6 / 8 / 71 | 15 / 6 / 8 / 71 |
| vs RAPTOR | Run 1 | 62 / 5 / 4 / 29 | 14 / 4 / 7 / 75 | 14 / 4 / 7 / 75 |
| vs RAPTOR | Run 2 | 63 / 9 / 3 / 25 | 12 / 5 / 9 / 74 | 12 / 5 / 9 / 74 |
| vs RAPTOR | Run 3 | 64 / 8 / 2 / 26 | 13 / 8 / 8 / 71 | 13 / 8 / 8 / 71 |

**Answer support, per run** (both / Agentic only / baseline only / neither)

|  | Run | Answer stated | Complete chain |
|---|---|---|---|
| vs RAG | Run 1 | 26 / 12 / 9 / 53 | 26 / 12 / 9 / 53 |
| vs RAG | Run 2 | 25 / 9 / 10 / 56 | 25 / 9 / 10 / 56 |
| vs RAG | Run 3 | 26 / 13 / 9 / 52 | 26 / 13 / 9 / 52 |
| vs HippoRAG | Run 1 | 29 / 9 / 11 / 51 | 29 / 9 / 11 / 51 |
| vs HippoRAG | Run 2 | 25 / 9 / 15 / 51 | 25 / 9 / 15 / 51 |
| vs HippoRAG | Run 3 | 29 / 10 / 11 / 50 | 29 / 10 / 11 / 50 |
| vs RAPTOR | Run 1 | 26 / 12 / 8 / 54 | 26 / 12 / 8 / 54 |
| vs RAPTOR | Run 2 | 24 / 10 / 10 / 56 | 24 / 10 / 10 / 56 |
| vs RAPTOR | Run 3 | 27 / 12 / 7 / 54 | 27 / 12 / 7 / 54 |

**The same question across the 3 runs** (all 3 / 2 / 1 / no run)

| Method | Found ≥50% | Found ≥75% | Found 100% | Support: answer stated | Support: complete chain |
|---|---|---|---|---|---|
| Agentic GraphRAG | 62 / 9 / 7 / 22 | 9 / 10 / 9 / 72 | 9 / 10 / 9 / 72 | 24 / 13 / 13 / 50 | 24 / 13 / 13 / 50 |
| RAG | 68 / 0 / 0 / 32 | 19 / 0 / 0 / 81 | 19 / 0 / 0 / 81 | 35 / 0 / 0 / 65 | 35 / 0 / 0 / 65 |
| HippoRAG | 63 / 0 / 0 / 37 | 23 / 0 / 0 / 77 | 23 / 0 / 0 / 77 | 40 / 0 / 0 / 60 | 40 / 0 / 0 / 60 |
| RAPTOR | 66 / 0 / 0 / 34 | 21 / 0 / 0 / 79 | 21 / 0 / 0 / 79 | 34 / 0 / 0 / 66 | 34 / 0 / 0 / 66 |

**Answers, per run** (both correct / Agentic only / baseline only / both wrong)

|  | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| vs RAG | 45 / 9 / 6 / 40 | 38 / 8 / 14 / 40 | 44 / 7 / 7 / 42 |
| vs HippoRAG | 38 / 16 / 8 / 38 | 35 / 11 / 13 / 41 | 39 / 12 / 11 / 38 |
| vs RAPTOR | 39 / 15 / 6 / 40 | 36 / 10 / 14 / 40 | 40 / 11 / 9 / 40 |

Answered right across the 3 runs (all 3 / 2 / 1 / no run): Agentic GraphRAG 35 / 18 / 10 / 37; RAG 46 / 7 / 2 / 45; HippoRAG 44 / 3 / 6 / 47; RAPTOR 43 / 3 / 9 / 45.

Wrong in every run (Agentic wrong ×3 & baseline right ×3 / baseline wrong ×3 & Agentic right ×3 / both wrong ×3): vs RAG 1 / 1 / 35; vs HippoRAG 2 / 7 / 32; vs RAPTOR 2 / 2 / 35.

## Temporal

**Gold-sentence recall, Agentic GraphRAG vs each baseline, per run** (both / Agentic only / baseline only / neither)

|  | Run | Found ≥50% | Found ≥75% | Found 100% |
|---|---|---|---|---|
| vs RAG | Run 1 | 47 / 17 / 5 / 31 | 13 / 2 / 2 / 83 | 13 / 2 / 2 / 83 |
| vs RAG | Run 2 | 47 / 15 / 5 / 33 | 13 / 7 / 2 / 78 | 13 / 7 / 2 / 78 |
| vs RAG | Run 3 | 48 / 8 / 4 / 40 | 13 / 2 / 2 / 83 | 13 / 2 / 2 / 83 |
| vs HippoRAG | Run 1 | 45 / 19 / 6 / 30 | 9 / 6 / 5 / 80 | 9 / 6 / 5 / 80 |
| vs HippoRAG | Run 2 | 45 / 17 / 6 / 32 | 10 / 10 / 4 / 76 | 10 / 10 / 4 / 76 |
| vs HippoRAG | Run 3 | 45 / 11 / 6 / 38 | 9 / 6 / 5 / 80 | 9 / 6 / 5 / 80 |
| vs RAPTOR | Run 1 | 45 / 19 / 3 / 33 | 12 / 3 / 4 / 81 | 12 / 3 / 4 / 81 |
| vs RAPTOR | Run 2 | 44 / 18 / 4 / 34 | 13 / 7 / 3 / 77 | 13 / 7 / 3 / 77 |
| vs RAPTOR | Run 3 | 46 / 10 / 2 / 42 | 12 / 3 / 4 / 81 | 12 / 3 / 4 / 81 |

**Answer support, per run** (both / Agentic only / baseline only / neither)

|  | Run | Answer stated | Complete chain |
|---|---|---|---|
| vs RAG | Run 1 | 32 / 8 / 4 / 56 | 32 / 8 / 4 / 56 |
| vs RAG | Run 2 | 29 / 12 / 7 / 52 | 29 / 11 / 7 / 53 |
| vs RAG | Run 3 | 31 / 6 / 5 / 58 | 31 / 6 / 5 / 58 |
| vs HippoRAG | Run 1 | 30 / 10 / 10 / 50 | 30 / 10 / 10 / 50 |
| vs HippoRAG | Run 2 | 26 / 15 / 14 / 45 | 26 / 14 / 14 / 46 |
| vs HippoRAG | Run 3 | 30 / 7 / 10 / 53 | 30 / 7 / 10 / 53 |
| vs RAPTOR | Run 1 | 28 / 12 / 5 / 55 | 28 / 12 / 5 / 55 |
| vs RAPTOR | Run 2 | 26 / 15 / 7 / 52 | 26 / 14 / 7 / 53 |
| vs RAPTOR | Run 3 | 27 / 10 / 6 / 57 | 27 / 10 / 6 / 57 |

**The same question across the 3 runs** (all 3 / 2 / 1 / no run)

| Method | Found ≥50% | Found ≥75% | Found 100% | Support: answer stated | Support: complete chain |
|---|---|---|---|---|---|
| Agentic GraphRAG | 48 / 11 / 16 / 25 | 13 / 3 / 5 / 79 | 13 / 3 / 5 / 79 | 27 / 11 / 15 / 47 | 27 / 11 / 14 / 48 |
| RAG | 52 / 0 / 0 / 48 | 15 / 0 / 0 / 85 | 15 / 0 / 0 / 85 | 36 / 0 / 0 / 64 | 36 / 0 / 0 / 64 |
| HippoRAG | 51 / 0 / 0 / 49 | 14 / 0 / 0 / 86 | 14 / 0 / 0 / 86 | 40 / 0 / 0 / 60 | 40 / 0 / 0 / 60 |
| RAPTOR | 48 / 0 / 0 / 52 | 16 / 0 / 0 / 84 | 16 / 0 / 0 / 84 | 33 / 0 / 0 / 67 | 33 / 0 / 0 / 67 |

**Answers, per run** (both correct / Agentic only / baseline only / both wrong)

|  | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| vs RAG | 40 / 16 / 7 / 37 | 41 / 13 / 10 / 36 | 43 / 15 / 4 / 38 |
| vs HippoRAG | 46 / 10 / 13 / 31 | 45 / 9 / 15 / 31 | 48 / 10 / 11 / 31 |
| vs RAPTOR | 43 / 13 / 5 / 39 | 40 / 14 / 10 / 36 | 44 / 14 / 5 / 37 |

Answered right across the 3 runs (all 3 / 2 / 1 / no run): Agentic GraphRAG 43 / 13 / 13 / 31; RAG 43 / 4 / 8 / 45; HippoRAG 56 / 3 / 4 / 37; RAPTOR 44 / 5 / 5 / 46.

Wrong in every run (Agentic wrong ×3 & baseline right ×3 / baseline wrong ×3 & Agentic right ×3 / both wrong ×3): vs RAG 2 / 5 / 24; vs HippoRAG 7 / 2 / 21; vs RAPTOR 2 / 6 / 25.

## Inference

**Gold-sentence recall, Agentic GraphRAG vs each baseline, per run** (both / Agentic only / baseline only / neither)

|  | Run | Found ≥50% | Found ≥75% | Found 100% |
|---|---|---|---|---|
| vs RAG | Run 1 | 39 / 6 / 5 / 50 | 9 / 6 / 2 / 83 | 8 / 4 / 3 / 85 |
| vs RAG | Run 2 | 34 / 8 / 10 / 48 | 6 / 4 / 5 / 85 | 5 / 3 / 6 / 86 |
| vs RAG | Run 3 | 35 / 7 / 9 / 49 | 7 / 2 / 4 / 87 | 6 / 1 / 5 / 88 |
| vs HippoRAG | Run 1 | 28 / 17 / 5 / 50 | 8 / 7 / 5 / 80 | 5 / 7 / 3 / 85 |
| vs HippoRAG | Run 2 | 25 / 17 / 8 / 50 | 4 / 6 / 9 / 81 | 3 / 5 / 5 / 87 |
| vs HippoRAG | Run 3 | 24 / 18 / 9 / 49 | 5 / 4 / 8 / 83 | 4 / 3 / 4 / 89 |
| vs RAPTOR | Run 1 | 37 / 8 / 7 / 48 | 10 / 5 / 3 / 82 | 9 / 3 / 3 / 85 |
| vs RAPTOR | Run 2 | 33 / 9 / 11 / 47 | 6 / 4 / 7 / 83 | 5 / 3 / 7 / 85 |
| vs RAPTOR | Run 3 | 33 / 9 / 11 / 47 | 7 / 2 / 6 / 85 | 6 / 1 / 6 / 87 |

**Answer support, per run** (both / Agentic only / baseline only / neither)

|  | Run | Answer stated | Complete chain |
|---|---|---|---|
| vs RAG | Run 1 | 74 / 12 / 5 / 9 | 17 / 8 / 5 / 70 |
| vs RAG | Run 2 | 73 / 9 / 6 / 12 | 12 / 13 / 10 / 65 |
| vs RAG | Run 3 | 69 / 8 / 10 / 13 | 17 / 10 / 5 / 68 |
| vs HippoRAG | Run 1 | 73 / 13 / 5 / 9 | 16 / 9 / 8 / 67 |
| vs HippoRAG | Run 2 | 71 / 11 / 7 / 11 | 13 / 12 / 11 / 64 |
| vs HippoRAG | Run 3 | 68 / 9 / 11 / 12 | 16 / 11 / 8 / 65 |
| vs RAPTOR | Run 1 | 77 / 9 / 6 / 8 | 17 / 8 / 5 / 70 |
| vs RAPTOR | Run 2 | 73 / 9 / 10 / 8 | 14 / 11 / 8 / 67 |
| vs RAPTOR | Run 3 | 68 / 9 / 15 / 8 | 16 / 11 / 6 / 67 |

**The same question across the 3 runs** (all 3 / 2 / 1 / no run)

| Method | Found ≥50% | Found ≥75% | Found 100% | Support: answer stated | Support: complete chain |
|---|---|---|---|---|---|
| Agentic GraphRAG | 34 / 7 / 13 / 46 | 7 / 2 / 9 / 82 | 6 / 1 / 7 / 86 | 67 / 18 / 8 / 7 | 17 / 6 / 14 / 63 |
| RAG | 44 / 0 / 0 / 56 | 11 / 0 / 0 / 89 | 11 / 0 / 0 / 89 | 79 / 0 / 0 / 21 | 22 / 0 / 0 / 78 |
| HippoRAG | 33 / 0 / 0 / 67 | 13 / 0 / 0 / 87 | 8 / 0 / 0 / 92 | 78 / 0 / 1 / 21 | 24 / 0 / 0 / 76 |
| RAPTOR | 44 / 0 / 0 / 56 | 13 / 0 / 0 / 87 | 12 / 0 / 0 / 88 | 83 / 0 / 0 / 17 | 22 / 0 / 0 / 78 |

**Answers, per run** (both correct / Agentic only / baseline only / both wrong)

|  | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| vs RAG | 94 / 2 / 1 / 3 | 95 / 2 / 1 / 2 | 95 / 1 / 2 / 2 |
| vs HippoRAG | 95 / 1 / 3 / 1 | 95 / 2 / 2 / 1 | 94 / 2 / 3 / 1 |
| vs RAPTOR | 95 / 1 / 3 / 1 | 96 / 1 / 2 / 1 | 95 / 1 / 3 / 1 |

Answered right across the 3 runs (all 3 / 2 / 1 / no run): Agentic GraphRAG 95 / 2 / 0 / 3; RAG 94 / 2 / 2 / 2; HippoRAG 97 / 0 / 1 / 2; RAPTOR 98 / 0 / 0 / 2.

Wrong in every run (Agentic wrong ×3 & baseline right ×3 / baseline wrong ×3 & Agentic right ×3 / both wrong ×3): vs RAG 0 / 0 / 2; vs HippoRAG 2 / 1 / 1; vs RAPTOR 2 / 1 / 1.

## Null

**Answers, per run** (both correct / Agentic only / baseline only / both wrong)

|  | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| vs RAG | 94 / 2 / 2 / 2 | 94 / 3 / 0 / 3 | 92 / 4 / 2 / 2 |
| vs HippoRAG | 94 / 2 / 1 / 3 | 92 / 5 / 1 / 2 | 92 / 4 / 1 / 3 |
| vs RAPTOR | 94 / 2 / 2 / 2 | 95 / 2 / 1 / 2 | 93 / 3 / 2 / 2 |

Answered right across the 3 runs (all 3 / 2 / 1 / no run): Agentic GraphRAG 93 / 5 / 0 / 2; RAG 93 / 1 / 3 / 3; HippoRAG 89 / 6 / 2 / 3; RAPTOR 95 / 1 / 0 / 4.

Wrong in every run (Agentic wrong ×3 & baseline right ×3 / baseline wrong ×3 & Agentic right ×3 / both wrong ×3): vs RAG 0 / 1 / 2; vs HippoRAG 0 / 0 / 2; vs RAPTOR 0 / 2 / 2.

## What the per-run view shows

- RAG, HippoRAG and RAPTOR give the same retrieval result in every run: on gold-sentence recall each question is found in all 3 runs or in none, and they get the same support verdict in all 3 runs on every question but 1. Only Agentic GraphRAG's retrieval changes from run to run.
- Agentic GraphRAG's result changes between runs on 16 comparison, 27 temporal, 20 inference questions (gold-sentence recall ≥50%) and on 26 comparison, 26 temporal, 26 inference questions (answer stated).
- Questions whose answer is supported with a complete chain in every run: comparison 24 for Agentic GraphRAG vs 34–40 for the baselines; temporal 27 for Agentic GraphRAG vs 33–40 for the baselines; inference 17 for Agentic GraphRAG vs 22–24 for the baselines.
- Within any single run, the Agentic-only and baseline-only support counts differ by at most 8 questions, so in one run Agentic GraphRAG is close to the baselines. What separates them is consistency: Agentic GraphRAG's supported questions move between runs, so fewer are supported in all three.

The same tables are on slides 11 (gold-sentence recall) and 12 (answer support) of the Experiments scene.

Warm Regards,
