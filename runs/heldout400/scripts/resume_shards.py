"""Resume a pass whose shards died on an infrastructure failure (Neo4j OOM, 2026-11-02).
Keeps every cleanly saved retrieval (no retrieval_error), answers them in a copy of
the run dir, and runs the missing questions as new shards. Appends to the manifest."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "runs/heldout400"
tag, method = sys.argv[1], sys.argv[2]
SHARDS = int(os.environ.get("SHARDS", "4"))
order = json.load(open(OUT / "questions.json"))["order"]
manifest = [json.loads(l) for l in open(OUT / "manifest.jsonl")]
dead = [m for m in manifest if m["tag"] == tag and m["method"] == method and m["exit"] != 0]
kept_dirs, done = [], set()
for m in dead:
    log = Path(m["log"])
    # The shard's run dir is the newest graphrag dir whose retrieval ids match its log.
    ids = [l.split("saved retrieval for ")[1].strip() for l in open(log) if "saved retrieval for" in l]
    cands = sorted((ROOT / "runs" / method).glob("2026*"), key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = next(c for c in cands if (c / "retrieval.jsonl").exists()
                   and ids and ids[0] in (c / "retrieval.jsonl").open().readline())
    good = [l for l in open(run_dir / "retrieval.jsonl") if l.strip() and not json.loads(l).get("retrieval_error")]
    new = run_dir.with_name(run_dir.name + f"-{tag}resume")
    new.mkdir(exist_ok=True)
    shutil.copy(run_dir / "settings.json", new / "settings.json")
    (new / "retrieval.jsonl").write_text("".join(good))
    done |= {int(json.loads(l)["record_id"].rsplit("-", 1)[1]) for l in good}
    kept_dirs.append((m["shard"], new))
print("kept", len(done), "clean retrievals; answering them", flush=True)
for shard, d in kept_dirs:
    subprocess.run([str(ROOT / ".venv/bin/python"), "-c", f"import benchmark; benchmark.answer_saved(__import__('pathlib').Path({str(d / 'retrieval.jsonl')!r}))"],
                   cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True, stdout=subprocess.DEVNULL)
    with open(OUT / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"tag": tag, "method": method, "shard": shard, "workers": SHARDS, "log": "resume-kept",
                            "exit": 0, "run_dir": str(d), "questions": sum(1 for _ in open(d / "retrieval.jsonl"))}) + "\n")
missing = [q for q in order if q not in done]
print("running", len(missing), "missing questions", flush=True)
sys.argv = ["heldout_run.py"]
import importlib.util
spec = importlib.util.spec_from_file_location("hr", ROOT / "runs/heldout400/scripts/heldout_run.py")
src = (ROOT / "runs/heldout400/scripts/heldout_run.py").read_text().split("passes = sys.argv")[0]
ns = {"__file__": str(ROOT / "runs/heldout400/scripts/heldout_run.py")}; exec(src, ns)
jobs = [ns["launch"](f"{tag}", method, missing[s::SHARDS], SHARDS, f"resume{s}") for s in range(SHARDS)]
ns["wait"](jobs)
