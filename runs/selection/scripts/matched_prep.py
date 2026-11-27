"""Build the retrieval files for the 27 Nov 2026 controls (runs/selection/SPEC.md). No model calls.

AGENT_FILL100 (rep 1-3): the agent's 10 items exactly as Agentic Hybrid RAG sent them in that repeat
  (runs/final500 graphrag, repeat r), then Normal RAG's top 100 by vector score (B4_100 repeat 1) in
  rank order, skipping chunks already present, until 100 items. Same budget as top 100 straight.
FORCED_TOP10 (rep 1-3): Normal RAG's top 10 (B4 repeat 1's first 10), answered with the forced-answer
  prompt (RECRAG_FORCED_ANSWER=1: no 'Insufficient Information').
FORCED_CLOSED (rep 1-3): no chunks, forced-answer prompt.

The answer step (benchmark.answer_saved) runs on the VM: runs/selection/scripts/vm_matched.sh.
Questions: the 250 comparison + temporal questions of runs/nograph/questions.json.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/scripts/matched_prep.py
Writes runs/selection/runs/matched/<VARIANT>_r<rep>/{settings.json,retrieval.jsonl,variant.json}.
"""
import json
from pathlib import Path

from benchmark import SavedRetrieval

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/selection/runs/matched"


def loc(d):
    d = str(d)
    return ROOT / d[d.index("runs/"):]


def load(dirs):
    return [SavedRetrieval.model_validate_json(line) for d in dirs for line in open(loc(d) / "retrieval.jsonl")
            if line.strip()]


J = json.loads((ROOT / "runs/judge_paper/runs.json").read_text())
top100 = load(J["B4_100"]["1"])
top50 = load(J["B4"]["1"])
assert len(top100) == len(top50) == 250
ids = [r.record_id for r in top100]
T100 = {r.record_id: r for r in top100}
T50 = {r.record_id: r for r in top50}
agent_reg = json.loads((ROOT / "runs/final500/run_registry.json").read_text())["multihop_rag"]["graphrag"]
settings = (loc(J["B4_100"]["1"][0]) / "settings.json").read_text()


def write(variant, rep, rows, k, note):
    d = OUT / f"{variant}_r{rep}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(settings)
    (d / "retrieval.jsonl").write_text("\n".join(r.model_dump_json() for r in rows) + "\n")
    (d / "variant.json").write_text(json.dumps({"variant": variant, "rep": rep, "passages_to_answer": k,
                                                "forced_answer": variant.startswith("FORCED"), "note": note}) + "\n")
    print(variant, rep, len(rows), "rows", d.relative_to(ROOT))


def with_passages(base, passages, scores, tool, k):
    r = base.retrieval.model_copy(update={"passages": passages, "scores": scores, "evidence": [],
                                          "trace": [{"tool": tool}]})
    return base.model_copy(update={"retrieval": r, "question_cost": None, "llm_usage": [],
                                   "options": base.options.model_copy(update={"top_k": k})})


stats = {}
for rep in "123":
    agent = {r.record_id: r for r in load(agent_reg[rep]) if r.record_id in T100}
    assert set(agent) == set(ids), rep
    rows, added = [], []
    for q in ids:
        a = agent[q].retrieval.passages
        assert len(a) == 10 and not agent[q].retrieval_error, q
        passages, scores = list(a), [None] * len(a)
        seen = set(a)
        for p, s in zip(T100[q].retrieval.passages, T100[q].retrieval.scores):
            if len(passages) == 100:
                break
            if p not in seen:
                passages.append(p); scores.append(s); seen.add(p)
        assert len(passages) == 100, q
        added.append(sum(p not in set(T100[q].retrieval.passages) for p in a))
        rows.append(with_passages(T100[q], passages, [s if s is not None else 0.0 for s in scores],
                                  "agent_10_then_top100_fill", 100))
    stats[rep] = round(sum(added) / len(added), 2)
    write("AGENT_FILL100", rep, rows, 100, "agent's 10 as sent, then top 100 fill; mean agent items not in top 100: "
          f"{stats[rep]}")
    write("FORCED_TOP10", rep, [with_passages(T50[q], T50[q].retrieval.passages[:10], T50[q].retrieval.scores[:10],
                                              "forced_top10", 10) for q in ids], 10, "Normal RAG top 10")
    write("FORCED_CLOSED", rep, [with_passages(T50[q], [], [], "forced_closed", 10) for q in ids], 0, "no chunks")
print("agent items outside top 100, mean per question:", stats)
