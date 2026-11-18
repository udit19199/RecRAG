"""Answer-support judge over the held-out 400 study, one (repeat, shard) pair at a time.

Each pair is 100 questions x 4 methods. Finished pairs have a DONE file and are skipped;
judge calls are cached, so a restart pays only for calls not made yet. Run 2's Agentic
GraphRAG shards were resumed after a crash; their two halves are in merged/r2-graphrag-sN.

Run from the repo root: nohup caffeinate -i .venv/bin/python runs/heldout400/scripts/heldout_support_judge.py &
"""
import json, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

OUT = Path("runs/heldout400/judge_support")
METHODS = ["rag", "raptor", "hipporag", "graphrag"]
dirs = defaultdict(dict)
for line in open("runs/heldout400/manifest.jsonl"):
    m = json.loads(line)
    if m["tag"] in ("r1", "r2", "r3") and str(m["shard"]).isdigit() and m["run_dir"] and m["questions"] == 100:
        dirs[(m["tag"], int(m["shard"]))][m["method"]] = m["run_dir"]
for s in range(4):
    dirs[("r2", s)]["graphrag"] = str(OUT / "merged" / f"r2-graphrag-s{s}")
pairs = sorted(dirs)
assert len(pairs) == 12 and all(len(dirs[p]) == 4 for p in pairs), dirs


def status(stage, err="none"):
    done = sum((OUT / f"{t}-s{s}" / "DONE").exists() for t, s in pairs)
    (OUT / "STATUS.md").write_text(
        f"# Answer-support judge, heldout400\n\nStage: {stage}\nProgress: {done}/12 (repeat, shard) pairs\n"
        f"Last error: {err}\nUpdated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


for t, s in pairs:
    dest = OUT / f"{t}-s{s}"
    if (dest / "DONE").exists():
        continue
    for attempt in range(1, 4):
        status(f"judging {t} shard {s} (attempt {attempt})")
        cmd = [sys.executable, "-m", "evals.retrieval_judge", *[dirs[(t, s)][m] for m in METHODS],
               "--output", str(dest), "--workers", "16", "--bootstrap", "1000"]
        with open(OUT / f"{t}-s{s}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (dest / "per_question.jsonl").exists():
            (dest / "DONE").touch()
            break
        status(f"{t} shard {s} attempt {attempt} failed", f"exit {rc}, see {t}-s{s}.log")
        time.sleep(30)
    else:
        status(f"gave up on {t} shard {s}", f"3 failed attempts, see {t}-s{s}.log")
        sys.exit(1)
status("all done")
