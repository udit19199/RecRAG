"""Held-out MultiHop-RAG study driver: all four methods on the same frozen questions.

Quality passes: each repeat runs all four methods at once, SHARDS worker
processes per method, so every method sees the same concurrency and the same
time window. Latency pass: the first LATENCY_N questions, one worker per method,
all four methods at once, so timing is not inflated by our own parallelism.
Writes runs/heldout400/manifest.jsonl (one line per finished shard)."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/heldout400"
Q = json.load(open(OUT / "questions.json"))["order"]
SHARDS = int(os.environ.get("SHARDS", "4"))
LATENCY_N = 40
PY = {"hipporag": str(ROOT / ".venv-hipporag/bin/python")}
METHODS = ["rag", "raptor", "hipporag", "graphrag"]

def launch(tag, method, indices, workers, shard):
    log = OUT / "logs" / f"{tag}_{method}_{shard}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    code = (f"import benchmark; p = benchmark.run({method!r}, 'multihop_rag', 0, {len(indices)}, "
            f"record_indices={indices!r}); print('RESULTS', p)")
    env = {**os.environ, "RECRAG_CONCURRENCY": str(workers), "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([PY.get(method, str(ROOT / ".venv/bin/python")), "-c", code],
                            cwd=ROOT, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    return dict(tag=tag, method=method, shard=shard, workers=workers, indices=indices, log=str(log), proc=proc)

def wait(jobs):
    while any(j["proc"].poll() is None for j in jobs):
        time.sleep(20)
    with open(OUT / "manifest.jsonl", "a") as f:
        for j in jobs:
            lines = [l for l in open(j["log"]) if l.startswith("RESULTS ")]
            f.write(json.dumps({k: j[k] for k in ("tag", "method", "shard", "workers", "log")}
                               | {"exit": j["proc"].returncode,
                                  "run_dir": str(Path(lines[-1].split(" ", 1)[1].strip()).parent) if lines else None,
                                  "questions": len(j["indices"])}) + "\n")
    failed = [j for j in jobs if j["proc"].returncode]
    print(time.strftime("%H:%M"), "finished", jobs[0]["tag"], "failed shards:", [(j["method"], j["shard"]) for j in failed], flush=True)

passes = sys.argv[1:] or ["latency", "r1", "r2", "r3"]
for tag in passes:
    if tag == "latency":
        jobs = [launch(tag, m, Q[:LATENCY_N], 1, 0) for m in METHODS]
    else:
        jobs = [launch(tag, m, Q[s::SHARDS], SHARDS, s) for m in METHODS for s in range(SHARDS)]
    wait(jobs)
