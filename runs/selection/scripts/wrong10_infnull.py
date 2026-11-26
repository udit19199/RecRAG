"""WRONG10 on the inference + null questions (runs/selection/SPEC.md, 26 Nov 2026). No retrieval call.

Each of the 250 inference/null questions is answered from the Normal RAG top 10 of ANOTHER question of the
same type, same answer model, prompt and settings as every other setup. Donors come from
runs/selection/paired/wrong10_infnull_donors.json, built on the Mac (seed 20261126): same question type,
no gold-article overlap (null questions have no gold, so any same-type donor is clean). The comparison/
temporal map wrong10_donors.json is not touched.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/scripts/wrong10_infnull.py REP [LIMIT]
Env: WRONG10_SRC = B0_infnull repeat-1 run dir (Normal RAG top 10 per question), default below.
Writes runs/selection/runs/rag/<stamp>Z/, prints "RESULTS <run dir>".
"""
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark import SavedRetrieval, answer_saved

ROOT = Path(__file__).resolve().parents[3]
rep = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None

src = ROOT / os.environ.get("WRONG10_SRC", "runs/selection/runs/rag/20261124T223730111524Z")
rows = [SavedRetrieval.model_validate_json(line) for line in open(src / "retrieval.jsonl")]
assert len(rows) == 250
by_id = {r.record_id: r for r in rows}
donors = json.loads((ROOT / "runs/selection/paired/wrong10_infnull_donors.json").read_text())

stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
run_dir = ROOT / "runs/selection/runs/rag" / stamp
run_dir.mkdir(parents=True)
(run_dir / "settings.json").write_text((src / "settings.json").read_text())
out = []
for q in rows[:limit]:
    d = by_id[donors[q.record_id]]
    r = q.retrieval.model_copy(update={"passages": d.retrieval.passages[:10],
                                       "scores": d.retrieval.scores[:10], "evidence": [],
                                       "trace": [{"tool": "wrong_chunks", "donor": d.record_id}]})
    out.append(q.model_copy(update={"retrieval": r, "question_cost": None, "llm_usage": [],
                                    "options": q.options.model_copy(update={"top_k": 10})}).model_dump_json())
(run_dir / "retrieval.jsonl").write_text("\n".join(out) + "\n")
path = answer_saved(run_dir / "retrieval.jsonl")
(run_dir / "variant.json").write_text(json.dumps({"variant": "WRONG10_infnull", "rep": rep, "limit": limit, "passages_to_answer": 10,
                                                  "donors": "runs/selection/paired/wrong10_infnull_donors.json",
                                                  "dataset": "multihop_rag"}) + "\n")
print("RESULTS", path, flush=True)
