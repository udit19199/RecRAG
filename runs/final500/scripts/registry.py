"""Write runs/final500/run_registry.json: every run folder of the study.

For each dataset, method and repeat, the run folders whose rows make up that
repeat. MultiHop-RAG Normal RAG, RAPTOR and HippoRAG repeats are the held-out
400 folders (runs/heldout400/manifest.jsonl, tags r1-r3, exit 0) plus this
study's 100 new questions; everything else is this study's shard folders.
Prints row counts per entry and refuses to write if any count is off.
"""

import json
from pathlib import Path

F = Path("runs/final500")
registry: dict[str, dict[str, dict[str, list[str]]]] = {}
problems = []
for dataset in ("multihop_rag", "hotpotqa", "natural_questions"):
    for method in ("rag", "raptor", "hipporag", "graphrag"):
        for rep in ("1", "2", "3"):
            dirs = sorted(
                str(p.resolve()) for p in (F / dataset / method / f"r{rep}").glob("shard*")
            )
            if dataset == "multihop_rag" and method != "graphrag":
                for line in open("runs/heldout400/manifest.jsonl"):
                    row = json.loads(line)
                    if row["tag"] == f"r{rep}" and row["method"] == method and row["exit"] == 0:
                        dirs.append(row["run_dir"])
            if not dirs:
                continue
            ids = [
                json.loads(line)["record_id"]
                for d in dirs
                for line in open(Path(d) / "results.jsonl")
            ]
            ok = len(ids) == 500 and len(set(ids)) == 500
            print(dataset, method, rep, len(dirs), "folders", len(ids), "rows", "ok" if ok else "BAD")
            if not ok:
                problems.append((dataset, method, rep, len(ids), len(set(ids))))
            registry.setdefault(dataset, {}).setdefault(method, {})[rep] = dirs
if problems:
    raise SystemExit(f"row counts off: {problems}")
(F / "run_registry.json").write_text(json.dumps(registry, indent=1))
