"""Score the 27 Nov controls (SPEC.md here): answered right per type and repeat, answer mix, and
paired 95% ranges (question by question, mean over repeats, 10,000 resamples, seed 1) against
top 100 straight (B4_100), Normal RAG (rag), the agent (graphrag) and closed book (CLOSED).
Same right rule as runs/selection/paired/review_checks.py (exact match, else the Haiku verdict).
No model calls.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/runs/matched/score.py
Writes runs/selection/runs/matched/scores.json.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "runs/selection/paired")
import review_checks as rc  # noqa: E402  (also rewrites review_checks.json, unchanged)
from evals.qa_metrics import normalize_answer  # noqa: E402

CT = ("comparison_query", "temporal_query")
QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
J = json.load(open("runs/judge_paper/runs.json"))
F = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]
REG = {k: J[k] for k in ("AGENT_FILL100", "FORCED_TOP10", "FORCED_CLOSED", "B4_100", "CLOSED")}
REG.update({"rag": F["rag"], "graphrag": F["graphrag"]})
ABST = normalize_answer("Insufficient Information")


def per_q(dirs):
    out = {}
    for r in rc.rows(dirs):
        if r["record_id"] in QS and r["question_type"] in CT:
            out[r["record_id"]] = (r["question_type"], rc.right(r), normalize_answer(r["answer"]) == ABST)
    return out


data = {k: {rep: per_q(d) for rep, d in v.items()} for k, v in REG.items()}
out = {"answered_right": {}, "answer_mix": {}, "paired": {}}
for k, reps in data.items():
    for qt in CT:
        pr = [round(100 * sum(x[1] for x in q.values() if x[0] == qt) / sum(x[0] == qt for x in q.values()), 1)
              for q in reps.values()]
        out["answered_right"].setdefault(k, {})[qt] = {"mean": round(sum(pr) / len(pr), 1), "repeats": pr}
        allq = [x for q in reps.values() for x in q.values() if x[0] == qt]
        n = len(allq)
        out["answer_mix"].setdefault(k, {})[qt] = {
            "right": round(100 * sum(x[1] for x in allq) / n, 1),
            "insufficient": round(100 * sum(x[2] for x in allq) / n, 1),
            "wrong": round(100 * sum(not x[1] and not x[2] for x in allq) / n, 1)}


def qmean(k, qt):
    reps = data[k].values()
    ids = {i for q in reps for i, x in q.items() if x[0] == qt}
    return {i: sum(q[i][1] for q in reps) / len(reps) for i in ids}


PAIRS = [("AGENT_FILL100", "B4_100"), ("AGENT_FILL100", "graphrag"), ("FORCED_TOP10", "rag"),
         ("FORCED_CLOSED", "CLOSED"), ("FORCED_TOP10", "FORCED_CLOSED"),
         ("B4_100", "FORCED_CLOSED"), ("graphrag", "FORCED_CLOSED"), ("AGENT_FILL100", "FORCED_CLOSED")]
for a, b in PAIRS:
    for qt in CT:
        A, B = qmean(a, qt), qmean(b, qt)
        d = [100 * (A[i] - B[i]) for i in sorted(A)]
        rng = random.Random(1)
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        out["paired"][f"{a} vs {b} {qt}"] = {"mean_diff_points": round(sum(d) / len(d), 1),
                                             "ci95": [round(boots[249], 1), round(boots[9749], 1)],
                                             "questions": len(d)}

# Forced answers: how many are 'Yes', and how often always-'Yes' would match them
for k in ("FORCED_TOP10", "FORCED_CLOSED"):
    for qt in CT:
        rows = [r for rep in REG[k].values() for r in rc.rows(rep) if r["record_id"] in QS and r["question_type"] == qt]
        out.setdefault("forced_yes_share", {}).setdefault(k, {})[qt] = round(
            100 * sum(normalize_answer(r["answer"]) == normalize_answer("Yes") for r in rows) / len(rows), 1)

Path("runs/selection/runs/matched/scores.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
