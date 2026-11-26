"""Every number the paper draft cites, gathered from the committed run data. No model calls.

Sources: runs/final500/report.json (four methods, all datasets, 3 repeats), runs/judge_paper/
slide_data.json (MultiHop-RAG comparison/temporal setups), runs/selection/paired/analysis.json
(answer mix, paired 95% ranges, inference/null), runs/selection/paired/cost.json (list price),
runs/selection/paired/review_checks.json (Natural Questions top K, always "Yes"),
runs/selection/SPEC.md (HotpotQA top K, 1 repeat). Tokens and model calls per question are
summed from results.jsonl llm_usage on the 250 comparison + temporal questions.

LLM-picker setups (B2, B2_100) and "agent picks, no graph rows" (A3) are left out on purpose.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/judge_paper/scripts/paper_numbers.py
-> runs/judge_paper/paper_numbers.json
"""
import ast
import json
from pathlib import Path
from statistics import mean, median

OUT = Path("runs/judge_paper/paper_numbers.json")
R = json.load(open("runs/final500/report.json"))
D = json.load(open("runs/judge_paper/slide_data.json"))
A = json.load(open("runs/selection/paired/analysis.json"))
C = json.load(open("runs/selection/paired/cost.json"))
RC = json.load(open("runs/selection/paired/review_checks.json"))
CT = ["comparison_query", "temporal_query"]
SKIP = {"B2", "B2_100", "B3", "A3_picks_no_graph_rows", "B2_milvus"}


def stat(v):
    if v is None or any(x is None for x in v):
        return None  # not scored
    return {"mean": round(mean(v), 2), "min": min(v), "max": max(v), "repeats": v}


out = {"four_methods": {}, "setups": {}, "answer_mix": {}, "paired": {}, "infnull": {},
       "cost_per_1000_list_usd": {}, "tokens": {}}

for ds, body in R.items():
    for qt, t in body["types"].items():
        for meas, m in t["measures"].items():
            n = m.get("questions") or t.get("questions")
            for meth, v in m["methods"].items():
                reps = v["per_repeat"]
                denom = n if meas == "Answered right" else v.get("questions", n)
                out["four_methods"].setdefault(ds, {}).setdefault(qt, {}).setdefault(meas, {})[meth] = {
                    "counts": reps, "n": denom, **stat([round(100 * x / denom, 1) for x in reps])}

for k, v in D.items():
    if k in SKIP:
        continue
    out["setups"][k] = {qt: {m: stat(v[qt][m]) for m in ("right", "f50", "f75", "f100") if m in v[qt]}
                        for qt in CT if qt in v}

out["answer_mix"] = {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in A["answer_mix"].items()
                     if k.split()[0].replace("_infnull", "") not in SKIP}
out["paired"] = {k: {"mean_diff": round(v["mean_diff_points"], 1), "ci95": [round(x, 1) for x in v["ci95"]],
                     "questions": v["questions"]}
                 for k, v in A["paired"].items() if not any(s.replace("_infnull", "") in SKIP for s in k.split())}
out["infnull"] = {"answer_mix": {k: {kk: round(vv, 1) for kk, vv in v.items()}
                                 for k, v in A["infnull"]["answer_mix"].items() if k.split()[0].replace("_infnull", "") not in SKIP},
                  "paired": {k: {"mean_diff": round(v["mean_diff_points"], 1),
                                 "ci95": [round(x, 1) for x in v["ci95"]]}
                             for k, v in A["infnull"]["paired"].items() if not any(s.replace("_infnull", "") in SKIP for s in k.split())}}
out["cost_per_1000_list_usd"] = {k: {qt: round(v[qt]["dollars_per_1000"]["mean"], 2) for qt in v}
                                 for k, v in C.items() if k not in SKIP}
out["review_checks"] = RC
out["hotpotqa_topk_1rep"] = {"B4_20": 87.8, "B4_30": 87.6, "B4": 87.4, "B4_100": 88.8, "B4_200": 89.4}


# tokens and model calls per question (comparison + temporal, the paired 250)
def loc(d):
    d = str(d)
    return Path(d[d.index("runs/"):])


QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
final = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]
REG = {"rag": final["rag"], "graphrag": final["graphrag"]}
REG.update({k: v for k, v in json.load(open("runs/judge_paper/runs.json")).items()
            if k in ("B4", "B4_100", "B4_200", "B4_20", "B4_30")})
REG["nograph"] = json.load(open("runs/nograph/run_registry.json"))  # {"1": [...], ...}
for m, reps in REG.items():
    tok, calls = [], []
    for rep in "123":
        for d in reps.get(rep, []):
            p = loc(d) / "results.jsonl"
            if not p.exists():
                continue
            for line in open(p):
                r = json.loads(line)
                if r["record_id"] not in QS or r["question_type"] not in CT:
                    continue
                u = r["llm_usage"]
                u = ast.literal_eval(u) if isinstance(u, str) else u
                tok.append(sum(x.get("total_tokens", 0) for x in u))
                calls.append(len(u))
    if tok:
        out["tokens"][m] = {"answers": len(tok), "median_tokens": median(tok), "mean_tokens": round(mean(tok)),
                            "mean_model_calls": round(mean(calls), 2)}

OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({"tokens": out["tokens"], "cost": out["cost_per_1000_list_usd"]}, indent=1))


# HotpotQA top K by question type (bridge / comparison), 1 repeat; same rule as review_checks.py
import sys  # noqa: E402
sys.path.insert(0, "runs/selection/paired")
import review_checks as rc  # noqa: E402  (re-runs its small checks; writes the same file)

hp = json.load(open("runs/selection/paired/datasets_registry.json"))["hotpotqa"]
out["hotpotqa_topk_by_type"] = {}
for k in ("B4_20", "B4_30", "B4", "B4_100", "B4_200"):
    rows = rc.rows(hp[k]["1"])
    for qt in sorted({r["question_type"] for r in rows}):
        sub = [r for r in rows if r["question_type"] == qt]
        out["hotpotqa_topk_by_type"].setdefault(k, {})[qt] = rc.pct(sub)
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out["hotpotqa_topk_by_type"]))
