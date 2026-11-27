"""Two checks from the 27 Nov 2026 deck review. No model calls; reads saved answers and verdicts.

1. Natural Questions, answered right with the runs/final500/scripts/report.py rule (exact match,
   else the Claude Haiku answer verdict; "Insufficient Information" is right only when a gold
   answer is "No answer"), for Normal RAG / Agentic Hybrid RAG (final500, 3 repeats) and the
   top K runs (runs/selection/paired/datasets_registry.json, 1 repeat). Slide 13 already used
   this rule; slide 14 had counted every abstention wrong.
2. Always "Yes": answered right if every MultiHop-RAG comparison / temporal question is
   answered "Yes", on the 250 questions of runs/nograph/questions.json (exact match against
   the gold answer).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/paired/review_checks.py
Writes runs/selection/paired/review_checks.json.
"""
import ast
import glob
import json
from pathlib import Path

from evals.qa_metrics import normalize_answer
from evals.subagent_judge import ABSTENTION, answer_key

OUT = Path("runs/selection/paired/review_checks.json")
NO_ANSWER = normalize_answer("No answer")
verdicts = {}
for p in glob.glob("runs/final500/judge/answers/verdicts-answers-*.json") + glob.glob(
        "runs/judge_paper/answers/verdicts-answers-*.json"):
    for v in json.load(open(p)):
        verdicts[v["id"]] = v["correct"]


def gold(row):
    g = row["expected_answers"]
    return ast.literal_eval(g) if isinstance(g, str) else g


def loc(d):
    d = str(d)
    return Path(d[d.index("runs/"):])


def rows(dirs):
    out = []
    for d in dirs:
        p = loc(d) / "results.jsonl"
        out += [json.loads(l) for l in open(p)]
    return out


def right(row):
    if float(row["exact_match"]) == 1:
        return True
    g = [normalize_answer(a) for a in gold(row)]
    a = normalize_answer(row["answer"])
    if a == ABSTENTION:
        return NO_ANSWER in g
    if not a or not g:
        return False
    return verdicts[answer_key(row["question"], row["answer"])]  # KeyError = unjudged answer


def pct(rs):
    return round(100 * sum(map(right, rs)) / len(rs), 2), len(rs)


nq = {}
f500 = json.load(open("runs/final500/run_registry.json"))["natural_questions"]
for m in ("rag", "graphrag"):
    nq[m] = {rep: pct(rows(dirs)) for rep, dirs in f500[m].items()}
sel = json.load(open("runs/selection/paired/datasets_registry.json"))["natural_questions"]
for k in ("B4_20", "B4_30", "B4", "B4_100", "B4_200"):
    nq[k] = {rep: pct(rows(dirs)) for rep, dirs in sel[k].items()}
no_answer_questions = sum(NO_ANSWER in [normalize_answer(a) for a in gold(r)]
                          for r in rows(sel["B4"]["1"]))

qs = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
mh = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]["rag"]["1"]
yes = {}
for r in rows(mh):
    if r["record_id"] not in qs or r["question_type"] not in ("comparison_query", "temporal_query"):
        continue
    t = yes.setdefault(r["question_type"], {"n": 0, "right": 0, "yes_or_no": 0})
    g = [normalize_answer(a) for a in gold(r)]
    t["n"] += 1
    t["right"] += normalize_answer("Yes") in g
    t["yes_or_no"] += bool({normalize_answer("Yes"), normalize_answer("No")} & set(g))
for t in yes.values():
    t["pct"] = round(100 * t["right"] / t["n"], 1)

res = {"natural_questions": nq, "nq_no_answer_questions": no_answer_questions, "always_yes": yes}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
