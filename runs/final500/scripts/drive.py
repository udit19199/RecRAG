"""Run answer jobs of the final 500-question study in parallel shards.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python runs/final500/scripts/drive.py NAME SHARDS JOB...
JOB is DATASET:METHOD:REP, e.g. hotpotqa:rag:1. Each job runs SHARDS worker
processes (answer.py), started a few seconds apart so run folders never share
a timestamp. Writes runs/final500/jobs/NAME/{STATUS.md,manifest.jsonl} and
links each shard's run folder as runs/final500/DATASET/METHOD/rREP/shardS.
STATUS.md says "stage: done" only after every job has the expected row count
in results.jsonl and no shard exited non-zero.
"""

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
F = ROOT / "runs/final500"
name, shards, jobs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
out = F / "jobs" / name
out.mkdir(parents=True, exist_ok=True)
questions = json.loads((F / "questions.json").read_text())
PY = {"hipporag": str(ROOT / ".venv-hipporag/bin/python")}


def expected(dataset: str, method: str) -> int:
    if dataset == "multihop_rag":
        return 500 if method == "graphrag" else 100
    return 500


def spend() -> str:
    return subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(F / "scripts/spend.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()


def status(stage: str, progress: str, error: str = "none") -> None:
    (out / "STATUS.md").write_text(
        f"stage: {stage}\nprogress: {progress}\nlast error: {error}\n"
        f"updated: {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        f"spend so far: {spend()}\n"
    )


procs = []
for job in jobs:
    dataset, method, rep = job.split(":")
    for shard in range(shards):
        log = out / f"{dataset}_{method}_r{rep}_s{shard}.log"
        proc = subprocess.Popen(
            [
                PY.get(method, str(ROOT / ".venv/bin/python")),
                str(F / "scripts/answer.py"),
                dataset,
                method,
                rep,
                str(shard),
                str(shards),
            ],
            cwd=ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
        )
        procs.append((dataset, method, rep, shard, log, proc))
        time.sleep(3)
status("running", f"{len(procs)} shards started")

total = sum(expected(d, m) for d, m, *_ in {(j.split(":")[0], j.split(":")[1], j.split(":")[2]) for j in jobs})
while any(p.poll() is None for *_, p in procs):
    saved = sum(
        log.read_text(errors="ignore").count("saved retrieval for")
        for *_, log, _ in procs
    )
    status("running", f"{saved} of {total} retrievals saved")
    time.sleep(120)

errors = []
with (out / "manifest.jsonl").open("a") as manifest:
    for dataset, method, rep, shard, log, proc in procs:
        lines = [x for x in log.read_text(errors="ignore").splitlines() if x.startswith("RESULTS ")]
        run_dir = Path(lines[-1].split(" ", 1)[1]).parent if lines else None
        if run_dir is not None:
            link = F / dataset / method / f"r{rep}" / f"shard{shard}"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                link.unlink()
            link.symlink_to(run_dir)
        manifest.write(
            json.dumps(
                {
                    "dataset": dataset,
                    "method": method,
                    "rep": rep,
                    "shard": shard,
                    "shards": shards,
                    "exit": proc.returncode,
                    "run_dir": str(run_dir) if run_dir else None,
                    "log": str(log),
                }
            )
            + "\n"
        )
        if proc.returncode:
            tail = log.read_text(errors="ignore").strip().splitlines()[-1:]
            errors.append(f"{dataset}/{method}/r{rep}/s{shard} exit {proc.returncode}: {tail}")

for job in jobs:
    dataset, method, rep = job.split(":")
    rows = 0
    for shard in range(shards):
        results = F / dataset / method / f"r{rep}" / f"shard{shard}" / "results.jsonl"
        if results.exists():
            rows += sum(1 for _ in results.open())
    if rows != expected(dataset, method):
        errors.append(f"{job}: {rows} rows, expected {expected(dataset, method)}")

if errors:
    status("failed", "row or exit check failed", " | ".join(errors)[:1500])
else:
    status("done", f"{len(jobs)} jobs, row counts checked")
