"""WRONG10: the "no relevant evidence" floor (runs/selection/SPEC.md, 25 Nov 2026). No retrieval call.

Each of the 250 questions is answered from the Normal RAG top 10 of ANOTHER question of the same type,
with the same answer model, prompt and settings as every other setup. A donor is accepted only if none of
its 10 chunks sits on an article that holds a gold sentence of the question, and none overlaps a gold
sentence. The donor map is fixed (seed 20261125) and the same in all 3 repeats, so the repeats differ only
in the answer call. Normal RAG's top 10 is B4 repeat 1's first 10 chunks (checked identical, 250/250,
in runs/selection/ceiling.json).

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/selection/scripts/wrong10.py REP [LIMIT]
Writes runs/selection/runs/rag/<stamp>Z/ (settings.json, retrieval.jsonl, results.jsonl, variant.json),
runs/selection/paired/wrong10_donors.json, prints "RESULTS <run dir>".
"""
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark import SavedRetrieval, answer_saved
from evals.retrieval_quality import any_overlap, load_corpus, locate_gold, ranked_items

ROOT = Path(__file__).resolve().parents[3]
rep = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
reg = json.loads((ROOT / "runs/selection/run_registry.json").read_text())["B4"]["1"][0]
src = ROOT / reg[reg.index("runs/"):]
rows = [SavedRetrieval.model_validate_json(line) for line in open(src / "retrieval.jsonl")]
assert len(rows) == 250
corpus = load_corpus("multihop_rag")


def top10(s: SavedRetrieval) -> SavedRetrieval:
    r = s.retrieval
    return s.model_copy(update={"retrieval": r.model_copy(update={"passages": r.passages[:10], "scores": r.scores[:10]})})


def pages(s: SavedRetrieval) -> set:
    return {corpus.page_of(sp.start) for it in ranked_items(top10(s), corpus) for sp in it.spans}


def clean(q: SavedRetrieval, donor: SavedRetrieval) -> bool:
    golds = locate_gold(q.supporting_sentences, corpus)
    assert golds, q.record_id
    if any(g.pages & pages(donor) for g in golds if g.pages):
        return False
    spans = [sp for it in ranked_items(top10(donor), corpus) for sp in it.spans]
    return not any(any_overlap(g, spans) for g in golds)


donor_path = ROOT / "runs/selection/paired/wrong10_donors.json"
if donor_path.exists():
    donors = json.loads(donor_path.read_text())
else:
    rng = random.Random(20261125)
    donors = {}
    for qt in sorted({r.question_type for r in rows}):
        group = [r for r in rows if r.question_type == qt]
        for q in group:
            order = group[:]
            rng.shuffle(order)
            tries = 0
            for d in order:
                tries += 1
                if d.record_id != q.record_id and clean(q, d):
                    donors[q.record_id] = d.record_id
                    break
            else:
                raise SystemExit(f"no clean donor for {q.record_id}")
    donor_path.parent.mkdir(parents=True, exist_ok=True)
    donor_path.write_text(json.dumps(donors, indent=1) + "\n")
by_id = {r.record_id: r for r in rows}

stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
run_dir = ROOT / "runs/selection/runs/rag" / stamp
run_dir.mkdir(parents=True)
(run_dir / "settings.json").write_text((src / "settings.json").read_text())
out = []
for q in rows[:limit]:
    d = by_id[donors[q.record_id]]
    r = q.retrieval.model_copy(update={"passages": d.retrieval.passages[:10], "scores": d.retrieval.scores[:10],
                                       "evidence": [], "trace": [{"tool": "wrong_chunks", "donor": d.record_id}]})
    out.append(q.model_copy(update={"retrieval": r, "question_cost": None, "llm_usage": [],
                                    "options": q.options.model_copy(update={"top_k": 10})}).model_dump_json())
(run_dir / "retrieval.jsonl").write_text("\n".join(out) + "\n")
path = answer_saved(run_dir / "retrieval.jsonl")
(run_dir / "variant.json").write_text(json.dumps({"variant": "WRONG10", "rep": rep, "limit": limit, "passages_to_answer": 10,
                                                  "donors": "runs/selection/paired/wrong10_donors.json"}) + "\n")
print("RESULTS", path, flush=True)
