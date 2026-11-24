"""Ceiling check for the top-k size curve (Bhanu, 24 Nov 2026). No model calls.

How often is all the gold evidence inside Normal RAG's top 10, 20, 30, 50?
Same text match as the retrieval-quality / evidence-funnel code
(evals/retrieval_quality.py: locate_gold + ranked_items + any_overlap): a gold
sentence counts as found when its text overlaps a retrieved passage's span in
the corpus. A gold sentence that cannot be located in the corpus counts as not
found (as in score_question).

Input: the B4 run folders (top 50 in retrieval order) from run_registry.json.
Repeats are checked to hold the same 50 passages per question.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/scripts/ceiling.py
  -> runs/selection/ceiling.json and a printed table.
"""
import json
from collections import defaultdict
from pathlib import Path

from benchmark import SavedRetrieval
from evals.retrieval_quality import any_overlap, load_corpus, locate_gold, ranked_items

KS = (10, 20, 30, 50)


def loc(d):
    """Registry paths are absolute on the machine that ran them; read from this checkout."""
    d = str(d)
    return Path(d[d.index("runs/"):])


reg = json.loads(Path("runs/selection/run_registry.json").read_text())["B4"]


def load(rep):
    rows = {}
    for d in reg[rep]:
        for line in open(loc(d) / "retrieval.jsonl"):
            s = SavedRetrieval.model_validate_json(line)
            rows[s.record_id] = s
    return rows


reps = {r: load(r) for r in sorted(reg)}
base = reps[min(reps)]
same = {r: sum(rows[q].retrieval.passages == base[q].retrieval.passages for q in base) for r, rows in reps.items()}

# Normal RAG's own top 10 (runs/final500) should be B4's first 10.
final = json.loads(Path("runs/final500/run_registry.json").read_text())["multihop_rag"]["rag"]
rag10 = {}
for d in final["1"]:
    for line in open(loc(d) / "retrieval.jsonl"):
        s = SavedRetrieval.model_validate_json(line)
        if s.record_id in base:
            rag10[s.record_id] = s.retrieval.passages
top10_same = sum(rag10[q] == base[q].retrieval.passages[:10] for q in rag10)

corpus = load_corpus("multihop_rag")
out = defaultdict(lambda: {"questions": 0, "gold_sentences": 0, "gold_unlocated": 0,
                           **{f"all_gold_top{k}": 0 for k in KS}, **{f"gold_found_top{k}": 0 for k in KS}})
for q, s in base.items():
    assert len(s.retrieval.passages) == 50, (q, len(s.retrieval.passages))
    items = ranked_items(s, corpus)
    golds = locate_gold(s.supporting_sentences, corpus)
    o = out[s.question_type]
    o["questions"] += 1
    o["gold_sentences"] += len(golds)
    o["gold_unlocated"] += sum(not g.spans for g in golds)
    for k in KS:
        spans = [sp for it in items[:k] for sp in it.spans]
        hit = [any_overlap(g, spans) for g in golds]
        o[f"gold_found_top{k}"] += sum(hit)
        o[f"all_gold_top{k}"] += bool(golds) and all(hit)

report = {"reps_same_50_passages": same, "questions": len(base),
          "top10_same_as_normal_rag_final500_rep1": f"{top10_same}/{len(rag10)}", "by_type": dict(out)}
Path("runs/selection/ceiling.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "by_type"}))
for qt, o in sorted(out.items()):
    n, g = o["questions"], o["gold_sentences"]
    print(f"{qt}: {n} questions, {g} gold sentences ({o['gold_unlocated']} not found in corpus)")
    for k in KS:
        print(f"  top {k:2}: all gold in {100 * o[f'all_gold_top{k}'] / n:5.1f}% of questions;"
              f" {100 * o[f'gold_found_top{k}'] / g:5.1f}% of gold sentences")
