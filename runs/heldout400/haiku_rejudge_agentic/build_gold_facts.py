"""Previous run's gold facts found, all judged by Claude Haiku (18 Nov 2026).

Writes runs/heldout400/gold_facts_by_run_haiku.json, same shape as gold_facts_by_run.json (whose gold facts were
judged by GPT-6 Luna and are kept unchanged for provenance):
  per_method[type][method][r1|r2|r3] = {"answered_right": <copied>, "0.5": n, "0.75": n, "1.0": n}
- graphrag: this folder's per_question.jsonl (Haiku re-judge of the previous run's Agentic passages, frozen facts
  from runs/final500/judge/claims, prompt 2026-11-10.haiku.2). Repeat = which r1/r2/r3 folder the run belongs to
  (runs/heldout400/manifest.jsonl; r2 = r2resume + 20261101T231404 folders).
- rag, raptor, hipporag: runs/final500/judge/per_question.jsonl rows whose run folder is under runs/heldout400
  (the latest run reused these exact previous-run outputs, so Haiku already judged them there).
Counts: questions whose share of gold facts found is >= 0.5 / 0.75 / 1.0, out of 100 per type.

Run from the repo root: python3 runs/heldout400/haiku_rejudge_agentic/build_gold_facts.py
"""
import json
from collections import defaultdict
from pathlib import Path

H = Path("runs/heldout400")
OLD = json.load(open(H / "gold_facts_by_run.json"))
TYPES = ["comparison_query", "temporal_query", "inference_query"]

rep_of = {}
for m in map(json.loads, open(H / "manifest.jsonl")):
    if m["run_dir"] and m["tag"] in ("r1", "r2", "r3"):
        rep_of[Path(m["run_dir"]).name] = m["tag"]

shares = defaultdict(dict)  # (type, method, rep) -> {record_id: share}
for line in open(H / "haiku_rejudge_agentic/per_question.jsonl"):
    r = json.loads(line)
    shares[(r["question_type"], r["method"], rep_of[Path(r["run"]).name])][r["record_id"]] = r["share"]
for line in open("runs/final500/judge/per_question.jsonl"):
    r = json.loads(line)
    if r["dataset"] == "multihop_rag" and r["method"] != "graphrag" and "/runs/heldout400/" in r["run"]:
        shares[(r["question_type"], r["method"], "r" + r["rep"])][r["record_id"]] = r["share"]

out = {k: OLD[k] for k in ("repeats", "questions")}
out["judge"] = "Claude Haiku, prompt 2026-11-10.haiku.2, frozen facts from runs/final500/judge/claims"
out["per_method"] = {}
for t in TYPES:
    out["per_method"][t] = {}
    for meth in ("rag", "raptor", "hipporag", "graphrag"):
        out["per_method"][t][meth] = {}
        for rep in ("r1", "r2", "r3"):
            s = shares[(t, meth, rep)]
            assert len(s) == OLD["questions"][t], (t, meth, rep, len(s))
            row = {"answered_right": OLD["per_method"][t][meth][rep]["answered_right"]}
            for k in (0.5, 0.75, 1.0):
                row[str(k)] = sum(v >= k - 1e-9 for v in s.values())
            out["per_method"][t][meth][rep] = row
out["per_method"]["null_query"] = OLD["per_method"]["null_query"]
(H / "gold_facts_by_run_haiku.json").write_text(json.dumps(out, indent=1))
for t in TYPES:
    for meth in ("rag", "graphrag"):
        print(t, meth, {rep: v for rep, v in out["per_method"][t][meth].items()})
