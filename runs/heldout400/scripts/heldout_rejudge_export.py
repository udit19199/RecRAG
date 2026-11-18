"""Export held-out evidence items that never got a strict-guide verdict.

Batch 2 used the earlier, looser evidence wording and was never re-judged, and
38 Agentic GraphRAG items have no verdict at all. Both are re-exported with the
current (strict) JUDGE_GUIDE into runs/heldout400/rejudge/ so the original
judge/ folder, which analysis.json is built from, stays unchanged."""
import glob, json
from pathlib import Path

from evals.subagent_judge import JudgeItems, evidence_items

OUT = Path("runs/heldout400")
latest = {}  # id -> verdict file that decides it (later files win, as in heldout_analysis)
for f in sorted(glob.glob(str(OUT / "judge" / "verdicts-*.json"))):
    for v in json.load(open(f)):
        latest[v["id"]] = Path(f).stem
manifest = [json.loads(l) for l in open(OUT / "manifest.jsonl")]
dirs = [Path(m["run_dir"]) for m in manifest if m["run_dir"] and m["tag"] != "latency"]
items = [e for e in evidence_items(dirs, "multihop_rag")
         if e.id not in latest or latest[e.id].startswith("verdicts-batch2-")]
dest = OUT / "rejudge"
dest.mkdir(exist_ok=True)
for n, start in enumerate(range(0, len(items), 60)):
    (dest / f"items-{n:03d}.json").write_text(JudgeItems(evidence=items[start:start + 60]).model_dump_json(indent=1))
print(f"{len(items)} items ({sum(e.id not in latest for e in items)} never judged) -> {dest}")
