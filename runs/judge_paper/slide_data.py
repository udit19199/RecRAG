"""Numbers for the "where does the gain come from" deck slides (22 Nov 2026).

Same 250 MultiHop-RAG questions (runs/nograph/questions.json: 125 comparison,
125 temporal), 3 repeats, % per repeat, per question type, never pooled.
- Answered right: exact match, else the Claude Haiku answer verdict
  (runs/final500, runs/nograph and runs/judge_paper answer verdict files).
- Gold facts found >=50% / >=75% / 100%: Claude Haiku verdicts
  (runs/final500/judge/per_question.jsonl for Normal RAG and the agent;
  runs/nograph/judge/per_question.jsonl for the agent with the graph off;
  runs/judge_paper/per_question.jsonl for the rest).

Usage (repo root, light): PYTHONPATH=. .venv/bin/python runs/judge_paper/slide_data.py
  -> runs/judge_paper/slide_data.json
"""
import glob
import json
from pathlib import Path

from evals.subagent_judge import _Result, answer_key, needs_answer_judge

QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
TYPES = ["comparison_query", "temporal_query"]
verd = {}
for p in (glob.glob("runs/final500/judge/answers/verdicts-answers-*.json")
          + glob.glob("runs/nograph/judge/answers/verdicts-answers-*.json")
          + glob.glob("runs/judge_paper/answers/verdicts-answers-*.json")):
    for v in json.load(open(p)):
        verd[v["id"]] = v["correct"]


def right(row):
    if row.get("exact_match") == 1:
        return True
    r = _Result.model_validate({k: row[k] for k in ("record_id", "question", "answer", "expected_answers", "exact_match")})
    if not needs_answer_judge(r):
        return False
    return verd[answer_key(row["question"], row["answer"])]


final = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]
REG = {"rag": final["rag"], "graphrag": final["graphrag"],
       "nograph": json.load(open("runs/nograph/run_registry.json"))}
REG.update(json.load(open("runs/judge_paper/runs.json")))
for name in ("B1",):  # added once their runs exist
    reg = json.load(open("runs/selection/run_registry.json"))
    if name in reg:
        REG[name] = reg[name]

run_rep = {d: (m, rep) for m, reps in REG.items() for rep, ds in reps.items() for d in ds}
facts = {}
for path, key in (("runs/final500/judge/per_question.jsonl", "final"), ("runs/nograph/judge/per_question.jsonl", "pilot"),
                  ("runs/judge_paper/per_question.jsonl", "paper")):
    for line in open(path):
        r = json.loads(line)
        if key == "final":
            if r["dataset"] != "multihop_rag" or r["method"] not in ("rag", "graphrag"):
                continue
            m, rep = r["method"], r["rep"]
        else:
            m, rep = (r["label"], r["rep"]) if key == "paper" else run_rep[r["run"]]
        if r["record_id"] in QS:
            facts.setdefault((m, rep, r["question_type"]), {})[r["record_id"]] = r["share"]

data = {}
for m, reps in REG.items():
    for qt in TYPES:
        out = {"right": [], "f50": [], "f75": [], "f100": [], "n": []}
        for rep in "123":
            rows = {}
            for d in reps[rep]:
                for line in open(Path(d) / "results.jsonl"):
                    r = json.loads(line)
                    if r["record_id"] in QS and r["question_type"] == qt:
                        rows[r["record_id"]] = r
            out["n"].append(len(rows))
            out["right"].append(100 * sum(map(right, rows.values())) / len(rows))
            # B4_rev / B4_shuf: B4's exact passages reordered, so B4's gold facts.
            sh = facts.get(({"B4_rev": "B4", "B4_shuf": "B4"}.get(m, m), rep, qt), {})
            for k, t in (("f50", 0.5), ("f75", 0.75), ("f100", 1.0)):
                out[k].append(100 * sum(s >= t - 1e-9 for s in sh.values()) / len(sh) if sh else None)
            out.setdefault("facts_n", []).append(len(sh))
        data.setdefault(m, {})[qt] = out

Path("runs/judge_paper/slide_data.json").write_text(json.dumps(data, indent=1))
for m, d in data.items():
    for qt, o in d.items():
        f = lambda v: "/".join("-" if x is None else f"{x:.0f}" for x in v)
        print(f"{m:24} {qt[:4]} n={o['n']} facts_n={o['facts_n']} right {f(o['right'])}  "
              f"facts {f(o['f50'])} | {f(o['f75'])} | {f(o['f100'])}")
