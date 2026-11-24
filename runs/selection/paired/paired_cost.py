"""Cost and latency for the deck's "What does each setup cost?" slide. No model calls.

Mean over the 125 questions of each type within a repeat, then the average and the
lowest-highest of the 3 repeats, per setup and question type (comparison, temporal).
Cost = results.jsonl question_cost calls estimated_usd (US$), reported in cents; latency =
retrieval_and_answer_seconds. Same rows and registries as analysis.py.

Usage (repo root, light): PYTHONPATH=. .venv/bin/python runs/selection/paired/paired_cost.py
  (not cost.py: run as a script it would shadow the repo-root cost.py for benchmark.py's imports)
  -> runs/selection/paired/cost.json
"""
import json
from pathlib import Path

OUT = Path("runs/selection/paired")
QS = {f"multihop-rag-{i}" for i in json.load(open("runs/nograph/questions.json"))["indices"]}
TYPES = ["comparison_query", "temporal_query"]
SETUPS = [("rag", "Normal RAG, top 10"), ("B2", "LLM picks 10 of top 50"),
          ("B4", "Top 50 straight"), ("B4_100", "Top 100 straight"),
          ("graphrag", "Agentic Hybrid RAG")]


def loc(d):
    d = str(d)
    return Path(d[d.index("runs/"):])


final = json.load(open("runs/final500/run_registry.json"))["multihop_rag"]
REG = {"rag": final["rag"], "graphrag": final["graphrag"]}
sel = json.load(open("runs/selection/run_registry.json"))
for name in sel:
    REG.setdefault(name, sel[name])
p = Path("runs/selection/paired/B4_100_registry.json")
if p.exists():
    REG["B4_100"] = json.load(open(p))["B4_100"]

rows = {}
for m, reps in REG.items():
    if m not in dict(SETUPS):
        continue
    for rep in "123":
        for d in reps.get(rep, []):
            for line in open(loc(d) / "results.jsonl"):
                r = json.loads(line)
                if r["record_id"] in QS and r["question_type"] in TYPES:
                    rows[(m, rep, r["record_id"])] = r

out = {}
lines = []
for m, name in SETUPS:
    if m not in REG:
        continue
    for qt in TYPES:
        qs = sorted({q for (mm, _, q) in rows if mm == m and rows[(mm, _, q)]["question_type"] == qt})
        cell = {}
        for field, key in (("question_cost", "cents"), ("retrieval_and_answer_seconds", "seconds")):
            per_rep = []
            for rep in "123":
                vals = [
                    (sum(c["estimated_usd"] for c in rows[(m, rep, q)][field]["calls"])
                     if field == "question_cost" else rows[(m, rep, q)][field])
                    for q in qs if (m, rep, q) in rows]
                per_rep.append(sum(vals) / len(vals))
            cell[key] = {"mean": sum(per_rep) / len(per_rep), "repeats": per_rep}
        if "cents" in cell:
            cell["cents"] = {"mean": cell["cents"]["mean"] * 100,
                             "repeats": [v * 100 for v in cell["cents"]["repeats"]]}
        out.setdefault(m, {})[qt] = cell
        lines.append(f"{m:10} {qt[:4]} cents {cell['cents']['mean']:.2f} "
                     f"({min(cell['cents']['repeats']):.2f}-{max(cell['cents']['repeats']):.2f})  "
                     f"seconds {cell['seconds']['mean']:.1f} "
                     f"({min(cell['seconds']['repeats']):.1f}-{max(cell['seconds']['repeats']):.1f})  n={len(qs)}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "cost.json").write_text(json.dumps(out, indent=1) + "\n")
print("\n".join(lines))
