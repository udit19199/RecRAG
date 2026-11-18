"""Tables of the final 500-question study (repo AGENTS.md reporting rules).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/final500/scripts/report.py DATASET...
Writes runs/final500/report.json and prints the tables.

Answered right: exact match; otherwise the Claude Haiku answer verdict for
answers that need one (subagent_judge.needs_answer_judge); abstentions are
wrong, except that on Natural Questions a gold "No answer" is matched by the
model's "Insufficient Information" (NQ 100 pilot rule). Gold facts found:
share of a question's frozen facts marked stated with a verified quote
(runs/final500/judge/per_question.jsonl); only questions with gold facts.
Split by question type, each repeat separately; no pooled numbers. Questions
that fail in all 3 repeats are listed. Every method is compared with Normal
RAG per type and repeat as (both, method only, Normal RAG only, neither).
Agentic Hybrid RAG is also reported on the questions never used in tuning
(tuning_questions.json). Closed-book: Answered right only. Context size:
median and largest tokens passed to the answer model per method.
"""

import glob
import json
import sys
from pathlib import Path
from statistics import median

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import ABSTENTION, _Result, answer_key, needs_answer_judge

F = Path("runs/final500")
METHODS = ["rag", "raptor", "hipporag", "graphrag"]
REPS = ["1", "2", "3"]
LEVELS = {"Gold facts ≥50%": 0.5, "Gold facts ≥75%": 0.75, "Gold facts 100%": 1.0}
registry = json.loads((F / "run_registry.json").read_text())
verdicts = {
    v["id"]: v["correct"]
    for path in glob.glob(str(F / "judge/answers/verdicts-answers-*.json"))
    for v in json.load(open(path))
}
shares = {}
for line in open(F / "judge/per_question.jsonl"):
    row = json.loads(line)
    shares[(row["dataset"], row["method"], row["rep"], row["record_id"])] = row["share"]
tuning = json.loads((F / "tuning_questions.json").read_text())
excluded = {
    i
    for part in tuning.values()
    if isinstance(part, dict)
    for ids in part.values()
    for i in ids
}


def right(row: dict, dataset: str) -> bool:
    if row.get("exact_match") == 1:
        return True
    gold = [normalize_answer(a) for a in row["expected_answers"]]
    answer = normalize_answer(row["answer"])
    if dataset == "natural_questions" and answer == ABSTENTION:
        return normalize_answer("No answer") in gold
    result = _Result.model_validate(
        {k: row[k] for k in ("record_id", "question", "answer", "expected_answers", "exact_match")}
    )
    if not needs_answer_judge(result):
        return False
    key = answer_key(row["question"], row["answer"])
    if key not in verdicts:
        raise SystemExit(f"no answer verdict for {row['record_id']} ({key})")
    return verdicts[key]


def kind(row: dict, dataset: str) -> str:
    return row.get("question_type") or ("all" if dataset == "natural_questions" else "?")


report: dict = {}
for dataset in sys.argv[1:]:
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for method in METHODS:
        for rep in REPS:
            rows[(method, rep)] = {
                r["record_id"]: r
                for d in registry[dataset][method][rep]
                for r in map(json.loads, open(Path(d) / "results.jsonl"))
            }
    closed = {
        rep: {r["record_id"]: r for r in map(json.loads, open(F / "closed_book" / dataset / f"r{rep}" / "results.jsonl"))}
        for rep in REPS
    }
    ids = sorted(rows[("rag", "1")])
    types = sorted({kind(rows[("rag", "1")][i], dataset) for i in ids})
    out: dict = {"context_tokens": {}, "types": {}}
    for method in METHODS:
        tokens = [rows[(method, rep)][i]["retrieved_tokens"] for rep in REPS for i in ids]
        out["context_tokens"][method] = {"median": median(tokens), "max": max(tokens)}
    for t in types:
        qs = [i for i in ids if kind(rows[("rag", "1")][i], dataset) == t]
        with_facts = [i for i in qs if (dataset, "rag", "1", i) in shares]
        measures = {"Answered right": (qs, lambda m, r, i: right(rows[(m, r)][i], dataset))}
        for label, level in LEVELS.items():
            measures[label] = (
                with_facts,
                lambda m, r, i, level=level: shares[(dataset, m, r, i)] >= level - 1e-9,
            )
        block: dict = {"questions": len(qs), "with_gold_facts": len(with_facts), "measures": {}}
        for label, (pool, hit) in measures.items():
            if not pool:
                continue
            entry: dict = {"questions": len(pool), "methods": {}}
            for method in METHODS:
                per = [sum(hit(method, r, i) for i in pool) for r in REPS]
                fail_all = [i for i in pool if not any(hit(method, r, i) for r in REPS)]
                vs_rag = [
                    [
                        sum(hit(method, r, i) and hit("rag", r, i) for i in pool),
                        sum(hit(method, r, i) and not hit("rag", r, i) for i in pool),
                        sum(hit("rag", r, i) and not hit(method, r, i) for i in pool),
                        sum(not hit(method, r, i) and not hit("rag", r, i) for i in pool),
                    ]
                    for r in REPS
                ]
                entry["methods"][method] = {
                    "per_repeat": per,
                    "fail_all_3": len(fail_all),
                    "fail_all_3_ids": fail_all,
                    "vs_normal_rag_per_repeat": vs_rag,
                }
            untuned = [i for i in pool if i not in excluded]
            entry["agentic_never_tuned"] = {
                "questions": len(untuned),
                "graphrag_per_repeat": [sum(hit("graphrag", r, i) for i in untuned) for r in REPS],
                "rag_per_repeat": [sum(hit("rag", r, i) for i in untuned) for r in REPS],
                "vs_normal_rag_per_repeat": [
                    [
                        sum(hit("graphrag", r, i) and hit("rag", r, i) for i in untuned),
                        sum(hit("graphrag", r, i) and not hit("rag", r, i) for i in untuned),
                        sum(hit("rag", r, i) and not hit("graphrag", r, i) for i in untuned),
                        sum(not hit("graphrag", r, i) and not hit("rag", r, i) for i in untuned),
                    ]
                    for r in REPS
                ],
            }
            if label == "Answered right":
                entry["closed_book_per_repeat"] = [
                    sum(right(closed[r][i], dataset) for i in pool) for r in REPS
                ]
            block["measures"][label] = entry
        out["types"][t] = block
    report[dataset] = out

old = json.loads((F / "report.json").read_text()) if (F / "report.json").exists() else {}
(F / "report.json").write_text(json.dumps({**old, **report}, indent=1, ensure_ascii=False))
for dataset, out in report.items():
    print(f"\n######## {dataset}  context tokens {out['context_tokens']}")
    for t, block in out["types"].items():
        print(f"\n== {t}: {block['questions']} questions, {block['with_gold_facts']} with gold facts")
        for label, e in block["measures"].items():
            cb = e.get("closed_book_per_repeat")
            print(f"  {label} (of {e['questions']})" + (f"  closed-book {cb}" if cb else ""))
            for m, v in e["methods"].items():
                print(f"    {m:9} {v['per_repeat']}  fail all 3: {v['fail_all_3']:3}  vs RAG {v['vs_normal_rag_per_repeat']}")
            u = e["agentic_never_tuned"]
            print(f"    never-tuned ({u['questions']}): agentic {u['graphrag_per_repeat']} rag {u['rag_per_repeat']} vs {u['vs_normal_rag_per_repeat']}")
