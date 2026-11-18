"""Re-answer the 2 Nov Agentic GraphRAG searches with plain passage text.

Separates the two 10 Nov changes. The searches are the old ones (no new
retrieval); only what the answer model reads changes: each vector_search
passage becomes its plain text (graphrag_retrieval.answer_text), as in the
10 Nov runs. cypher_search rows are sent as before. Comparison questions only.

Writes runs/agentic_tune/reanswer/<repeat>/ (retrieval.jsonl, settings.json,
results.jsonl). Run from the repo root:
PYTHONPATH=. uv run python runs/heldout400/scripts/agentic_reanswer_plain.py"""

import json
import shutil
from pathlib import Path

import benchmark
from graphrag_retrieval import answer_text

H = Path("runs/heldout400")
OUT = Path("runs/agentic_tune/reanswer")
dirs: dict[str, list[Path]] = {}
for line in open(H / "manifest.jsonl"):
    m = json.loads(line)
    if m["method"] == "graphrag" and m["tag"] in ("r1", "r2", "r3") and m["exit"] == 0 and m["run_dir"]:
        d = Path(m["run_dir"])
        dirs.setdefault(m["tag"], []).append(d if d.is_absolute() else H / d)

for tag, run_dirs in sorted(dirs.items()):
    out = OUT / tag
    if (out / "results.jsonl").exists():
        print(tag, "already answered")
        continue
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dirs[0] / "settings.json", out / "settings.json")
    rows = []
    for d in run_dirs:
        for line in open(d / "retrieval.jsonl"):
            saved = json.loads(line)
            if saved["question_type"] != "comparison_query":
                continue
            saved["retrieval"]["passages"] = [answer_text(p) for p in saved["retrieval"]["passages"]]
            rows.append(saved)
    assert len(rows) == 100, (tag, len(rows))
    (out / "retrieval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(tag, benchmark.answer_saved(out / "retrieval.jsonl"))
