"""Numbers for two deck slides (18 Nov 2026), answered right only, % per repeat.
1. nograph: latest run, comparison and temporal, 125 questions: Normal RAG, Agentic Hybrid RAG, Agentic with no graph.
   Also wrong "Insufficient Information" answers (% of questions).
2. split: latest run on the 100 questions per type shared with the previous run, and on the 25 new ones;
   previous run on the same 100 (runs/heldout400/gold_facts_by_run.json answered_right).
Answered right = exact match, else the Haiku answer verdict (runs/final500 and runs/nograph verdict files).
Usage (repo root): PYTHONPATH=. .venv/bin/python runs/nograph/slide_data.py  -> runs/nograph/slide_data.json
"""
import glob, json
from pathlib import Path
from evals.qa_metrics import normalize_answer
from evals.subagent_judge import _Result, answer_key, needs_answer_judge

F = Path("runs/final500")
verd = {v["id"]: v["correct"] for p in glob.glob(str(F/"judge/answers/verdicts-answers-*.json")) + glob.glob("runs/nograph/judge/answers/verdicts-answers-*.json") for v in json.load(open(p))}
def right(row):
    if row.get("exact_match") == 1: return True
    r = _Result.model_validate({k: row[k] for k in ("record_id","question","answer","expected_answers","exact_match")})
    if not needs_answer_judge(r): return False
    return verd[answer_key(row["question"], row["answer"])]
def insuff(row): return normalize_answer(row["answer"]).startswith("insufficient")
reg = json.load(open(F/"run_registry.json"))["multihop_rag"]
reg["nograph"] = json.load(open("runs/nograph/run_registry.json"))
q = json.load(open(F/"questions.json"))["multihop_rag"]
new = {f"multihop-rag-{i}" for i in q["new100"]}
def rows(method, rep):
    out = {}
    for d in reg[method][rep]:
        for l in open(Path(d)/"results.jsonl"):
            r = json.loads(l); out[r["record_id"]] = r
    return out
data = {"nograph": {}, "split": {}}
TYPES = ["comparison_query", "temporal_query", "inference_query", "null_query"]
for qt in TYPES:
    for m in ["rag", "graphrag", "raptor", "hipporag", "nograph"]:
        if m == "nograph" and qt not in TYPES[:2]: continue
        per = {"all": [], "shared100": [], "new25": [], "insuff_all": []}
        for rep in "123":
            rr = [r for r in rows(m, rep).values() if r["question_type"] == qt]
            if m == "nograph" or True:
                pass
            for k, sel in [("all", rr), ("shared100", [r for r in rr if r["record_id"] not in new]), ("new25", [r for r in rr if r["record_id"] in new])]:
                per[k].append(100 * sum(map(right, sel)) / len(sel) if sel else None)
            per["insuff_all"].append(100 * sum(map(insuff, rr)) / len(rr))
            per.setdefault("n", {"all": len(rr), "shared100": sum(r["record_id"] not in new for r in rr), "new25": sum(r["record_id"] in new for r in rr)})
        if qt in TYPES[:2] and m in ("rag", "graphrag", "nograph"):
            data["nograph"].setdefault(qt, {})[m] = {"right": per["all"], "insuff": per["insuff_all"], "n": per["n"]["all"]}
        if m != "nograph":
            data["split"].setdefault(qt, {})[m] = {"shared100": per["shared100"], "new25": per["new25"], "n": per["n"]}
PD = json.load(open("runs/heldout400/gold_facts_by_run.json"))
for qt in TYPES:
    for m in ["rag", "graphrag", "raptor", "hipporag"]:
        data["split"][qt][m]["previous100"] = [PD["per_method"][qt][m][r]["answered_right"] * 100 / PD["questions"][qt] for r in ("r1","r2","r3")]
Path("runs/nograph/slide_data.json").write_text(json.dumps(data, indent=1))
avg = lambda v: sum(v)/3
for qt, d in data["nograph"].items():
    print(qt, {m: (round(avg(x["right"]),1), [round(v,1) for v in x["right"]], "insuff", round(avg(x["insuff"]),1), x["n"]) for m, x in d.items()})
for qt, d in data["split"].items():
    print(qt)
    for m, x in d.items():
        print(f"  {m:9} prev100 {avg(x['previous100']):5.1f} {[round(v) for v in x['previous100']]}  latest100 {avg(x['shared100']):5.1f} {[round(v) for v in x['shared100']]}  new25 {avg(x['new25']):5.1f} {[round(v) for v in x['new25']]}  n={x['n']}")
