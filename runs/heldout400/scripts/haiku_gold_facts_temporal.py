"""Temporal questions: gold facts found (Claude Haiku judge), per repeat and as one
2-of-3 confusion matrix per measure, for Normal RAG, Agentic GraphRAG 2 Nov and
Agentic GraphRAG 10 Nov. Matrix cells: [both, Agentic only, Normal RAG only, neither];
a question counts as found when the method reaches the threshold in at least 2 of 3
repeats. Run from the repo root:
PYTHONPATH=. uv run python runs/heldout400/scripts/haiku_gold_facts_temporal.py"""
import json
from pathlib import Path

H = Path("runs/heldout400")
J = Path("runs/agentic_tune/fact_judge_temporal")
TAGS = ("r1", "r2", "r3")
dirs = {}
for line in open(H / "manifest.jsonl"):
    m = json.loads(line)
    if m["method"] in ("rag", "graphrag") and m["tag"] in TAGS and m["exit"] == 0 and m["run_dir"]:
        d = m["run_dir"]
        dirs[str(Path(d) if d.startswith("/") else H / d)] = ("rag" if m["method"] == "rag" else "old", m["tag"])
for tag, name in zip(TAGS, ("temporal100", "temporal100_rep2", "temporal100_rep3")):
    dirs[f"runs/agentic_tune/graphrag/{name}"] = ("new", tag)
share = {}
for line in open(J / "per_question.jsonl"):
    r = json.loads(line)
    share[dirs[r["run"]] + (r["record_id"],)] = r["share"]
qs = sorted({k[2] for k in share})
assert len(qs) == 100 and all((m, t, q) in share for m in ("rag", "old", "new") for t in TAGS for q in qs)
out = {}
for label, th in ((">=50%", 0.5), (">=75%", 0.75), ("100%", 1.0)):
    hit = {(m, q): sum(share[(m, t, q)] >= th - 1e-9 for t in TAGS) for m in ("rag", "old", "new") for q in qs}
    row = {}
    for m in ("rag", "old", "new"):
        per = [sum(share[(m, t, q)] >= th - 1e-9 for q in qs) for t in TAGS]
        row[m] = {"per_repeat": per, "avg": round(sum(per) / 3), "lo": min(per), "hi": max(per)}
    b = [hit[("rag", q)] >= 2 for q in qs]
    for m in ("old", "new"):
        a = [hit[(m, q)] >= 2 for q in qs]
        row[m]["matrix_2of3"] = [sum(x and y for x, y in zip(a, b)), sum(x and not y for x, y in zip(a, b)),
                                 sum(y and not x for x, y in zip(a, b)), sum(not x and not y for x, y in zip(a, b))]
    out[label] = row
    print(label, {m: (row[m]["avg"], row[m]["lo"], row[m]["hi"], row[m].get("matrix_2of3")) for m in row})
(J / "temporal_final.json").write_text(json.dumps(out, indent=1))
