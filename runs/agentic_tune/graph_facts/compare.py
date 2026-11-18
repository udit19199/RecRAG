"""10 Nov version vs 13 Nov agent picks vs 13 Nov graph facts (final), on the 38
swinging comparison questions. Run from repo root:
PYTHONPATH=. uv run python runs/agentic_tune/graph_facts/compare.py"""
import json, glob
from pathlib import Path
from statistics import median
from evals.subagent_judge import answer_key
A = Path("runs/agentic_tune")
Q = json.loads((A / "graph_facts/questions.json").read_text())["record_ids"]
RUNS = {
    "10nov": {"r1": "graphrag/comparison100", "r2": "graphrag/comparison100_rep2", "r3": "graphrag/comparison100_rep3"},
    "picks": {"r1": "agent_picks/rep1", "r2": "agent_picks/rep2", "r3": "agent_picks/rep3"},
    "facts": {"r1": "graph_facts/rep1", "r2": "graph_facts/rep2", "r3": "graph_facts/rep3"},
}
REPS = ("r1", "r2", "r3")
resolved = {(t, rep): str((A / d).resolve()) for t, runs in RUNS.items() for rep, d in runs.items()}
share = {}
for l in open(A / "fact_judge/per_question.jsonl"):
    r = json.loads(l)
    run = str(Path(r["run"]).resolve()) if Path(r["run"]).exists() else r["run"]
    for (t, rep), d in resolved.items():
        if run == d or r["run"].endswith(RUNS[t][rep]):
            share[(t, rep, r["record_id"])] = r["share"]
verd = {}
for f in sorted(glob.glob("runs/heldout400/judge/verdicts-*.json")) + sorted(glob.glob(str(A / "judge/verdicts-*.json"))):
    for v in json.load(open(f)):
        verd[v["id"]] = v
right, tokens = {}, {}
for t, runs in RUNS.items():
    for rep, d in runs.items():
        for l in open(A / d / "results.jsonl"):
            r = json.loads(l)
            if r["record_id"] not in Q: continue
            right[(t, rep, r["record_id"])] = bool(r["exact_match"]) or bool(verd.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
            tokens.setdefault(t, []).append(r["retrieved_tokens"])
missing = [(t, rep, q) for t in RUNS for rep in REPS for q in Q if (t, rep, q) not in share or (t, rep, q) not in right]
assert not missing, missing[:5]
measures = {
    "Answered right": lambda t, rep, q: right[(t, rep, q)],
    "Gold facts ≥50%": lambda t, rep, q: share[(t, rep, q)] >= 0.5,
    "Gold facts ≥75%": lambda t, rep, q: share[(t, rep, q)] >= 0.75,
    "Gold facts 100%": lambda t, rep, q: share[(t, rep, q)] >= 0.999,
}
out = {"context_tokens": {t: {"median": median(v), "max": max(v)} for t, v in tokens.items()}}
for name, f in measures.items():
    row = {}
    for t in RUNS:
        per = [sum(f(t, rep, q) for q in Q) for rep in REPS]
        all3 = sum(all(f(t, rep, q) for rep in REPS) for q in Q)
        none3 = sum(not any(f(t, rep, q) for rep in REPS) for q in Q)
        row[t] = {"per_repeat": per, "in_all_3": all3, "in_none": none3}
    two = lambda t, q: sum(f(t, rep, q) for rep in REPS) >= 2
    for other in ("10nov", "picks"):
        row[f"matrix_2of3_facts_vs_{other}"] = [
            sum(two("facts", q) and two(other, q) for q in Q),
            sum(two("facts", q) and not two(other, q) for q in Q),
            sum(two(other, q) and not two("facts", q) for q in Q),
            sum(not two("facts", q) and not two(other, q) for q in Q)]
    out[name] = row
    print(f"\n{name}")
    for t in RUNS:
        r = row[t]; print(f"  {t:6} per repeat {r['per_repeat']}  all 3: {r['in_all_3']:2}  none: {r['in_none']:2}")
    for other in ("10nov", "picks"):
        print(f"  facts vs {other} [both, only facts, only {other}, neither]:", row[f"matrix_2of3_facts_vs_{other}"])
print("\ncontext tokens", out["context_tokens"])
(A / "graph_facts/compare.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
