"""Comparison questions: Answered right and gold facts found (Claude Haiku judge),
per repeat, for Normal RAG, old Agentic GraphRAG (2 Nov) and new Agentic GraphRAG
(10 Nov fixes). Pair cells: [both, Agentic only, Normal RAG only, neither] against
Normal RAG of the same repeat. Run from the repo root with PYTHONPATH=."""
import glob
import json
from pathlib import Path

from evals.subagent_judge import answer_key

H = Path("runs/heldout400")
NEW = {"r1": "runs/agentic_tune/graphrag/comparison100", "r2": "runs/agentic_tune/graphrag/comparison100_rep2",
       "r3": "runs/agentic_tune/graphrag/comparison100_rep3"}
dirs = {}
for line in open(H / "manifest.jsonl"):
    m = json.loads(line)
    if m["method"] in ("rag", "graphrag") and m["tag"] in NEW and m["exit"] == 0 and m["run_dir"]:
        d = m["run_dir"]
        dirs[str(Path(d) if d.startswith("/") else H / d)] = ("rag" if m["method"] == "rag" else "old", m["tag"])
for tag, d in NEW.items():
    dirs[d] = ("new", tag)
verdicts = {}
for f in sorted(glob.glob(str(H / "judge/verdicts-*.json"))) + sorted(glob.glob("runs/agentic_tune/judge/verdicts-*.json")):
    for v in json.load(open(f)):
        verdicts[v["id"]] = v
right, share = {}, {}
for d, key in dirs.items():
    for line in open(Path(d) / "results.jsonl"):
        r = json.loads(line)
        if r["question_type"] == "comparison_query":
            right[key + (r["record_id"],)] = bool(r["exact_match"]) or bool(verdicts.get(answer_key(r["question"], r["answer"]), {}).get("correct"))
for line in open("runs/agentic_tune/fact_judge/per_question.jsonl"):
    r = json.loads(line)
    share[dirs[r["run"]] + (r["record_id"],)] = r["share"]
qs = sorted({k[2] for k in share})
assert len(qs) == 100 and all((m, t, q) in share and (m, t, q) in right for m in ("rag", "old", "new") for t in NEW for q in qs)
MEASURES = [("answered right", None), (">=50%", 0.5), (">=75%", 0.75), ("100%", 1.0)]
def ok(m, t, q, th): return right[(m, t, q)] if th is None else share[(m, t, q)] >= th - 1e-9
out = {}
for label, th in MEASURES:
    for t in NEW:
        b = [ok("rag", t, q, th) for q in qs]
        row = {"rag": sum(b)}
        for m in ("old", "new"):
            a = [ok(m, t, q, th) for q in qs]
            row[m] = sum(a)
            row[m + "_cells"] = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                                 sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
        out[f"{label}|{t}"] = row
        print(f"{label:15s} {t}  new {row['new']:3d} {row['new_cells']}  old {row['old']:3d} {row['old_cells']}  rag {row['rag']:3d}")
for label, th in MEASURES:
    fails = [q for q in qs if all(not ok("new", t, q, th) for t in NEW)], [q for q in qs if all(not ok("rag", t, q, th) for t in NEW)]
    both = len(set(fails[0]) & set(fails[1]))
    print(f"fails in all 3 repeats, {label}: new Agentic {len(fails[0])}, Normal RAG {len(fails[1])}, both {both}")
Path("runs/agentic_tune/fact_judge/comparison_final.json").write_text(json.dumps(out, indent=1))
