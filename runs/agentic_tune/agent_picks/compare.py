"""Old (10 Nov) vs new (13 Nov agent picks) Agentic GraphRAG on the 38 swinging
comparison questions. Run from repo root:
PYTHONPATH=. uv run python runs/agentic_tune/agent_picks/compare.py"""
import json, glob
from pathlib import Path
from evals.subagent_judge import answer_key
A = Path("runs/agentic_tune")
Q = json.loads((A / "agent_picks/questions.json").read_text())["record_ids"]
OLD = {"r1": "graphrag/comparison100", "r2": "graphrag/comparison100_rep2", "r3": "graphrag/comparison100_rep3"}
NEW = {"r1": "agent_picks/rep1", "r2": "agent_picks/rep2", "r3": "agent_picks/rep3"}
share = {}
for l in open(A / "fact_judge/per_question.jsonl"):
    r = json.loads(l)
    for tag, runs in (("old", OLD), ("new", NEW)):
        for rep, d in runs.items():
            if r["run"].endswith(d):
                share[(tag, rep, r["record_id"])] = r["share"]
verd = {}
for f in sorted(glob.glob("runs/heldout400/judge/verdicts-*.json")) + sorted(glob.glob(str(A / "judge/verdicts-*.json"))):
    for v in json.load(open(f)):
        verd[v["id"]] = v
right = {}
for tag, runs in (("old", OLD), ("new", NEW)):
    for rep, d in runs.items():
        for l in open(A / d / "results.jsonl"):
            r = json.loads(l)
            right[(tag, rep, r["record_id"])] = bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
measures = {
    "Answered right": lambda t, rep, q: right[(t, rep, q)],
    "Gold facts ≥50%": lambda t, rep, q: share[(t, rep, q)] >= 0.5,
    "Gold facts ≥75%": lambda t, rep, q: share[(t, rep, q)] >= 0.75,
    "Gold facts 100%": lambda t, rep, q: share[(t, rep, q)] >= 0.999,
}
out = {}
for name, f in measures.items():
    row = {}
    for t in ("old", "new"):
        per = [sum(f(t, rep, q) for q in Q) for rep in ("r1", "r2", "r3")]
        all3 = sum(all(f(t, rep, q) for rep in ("r1", "r2", "r3")) for q in Q)
        none3 = sum(not any(f(t, rep, q) for rep in ("r1", "r2", "r3")) for q in Q)
        mixed = len(Q) - all3 - none3
        row[t] = {"per_repeat": per, "in_all_3": all3, "in_none": none3, "mixed": mixed}
    two = lambda t, q: sum(f(t, rep, q) for rep in ("r1", "r2", "r3")) >= 2
    row["matrix_2of3_new_vs_old"] = [sum(two("new", q) and two("old", q) for q in Q), sum(two("new", q) and not two("old", q) for q in Q),
                                     sum(two("old", q) and not two("new", q) for q in Q), sum(not two("new", q) and not two("old", q) for q in Q)]
    out[name] = row
    print(f"\n{name}")
    for t in ("old", "new"):
        r = row[t]; print(f"  {t}: per repeat {r['per_repeat']}  in all 3: {r['in_all_3']}  in none: {r['in_none']}  mixed: {r['mixed']}")
    print("  2-of-3 matrix [both, only new, only old, neither]:", row["matrix_2of3_new_vs_old"])
(A / "agent_picks/compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
