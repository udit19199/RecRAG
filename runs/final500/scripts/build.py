"""Build one index of the final 500-question study, with no study question.

Usage (repo root; HippoRAG with .venv-hipporag/bin/python):
  PYTHONPATH=. .venv/bin/python runs/final500/scripts/build.py DATASET METHOD

DATASET is hotpotqa or natural_questions; METHOD is rag, raptor, hipporag or
graphrag. The index goes where benchmark.run will look for it (namespace
final500/DATASET). The build is triggered by the benchmark's own warm-up
question, which is outside every question set and unscored, exactly as at the
start of a benchmark run. Build calls are metered and, with
RECRAG_BUILD_SERVICE_TIER=flex, sent at the flex tier (cost.py).
Writes runs/final500/DATASET/METHOD/build/{build.json,build_cost.json}.
"""

import json
import os
import sqlite3
import sys
import time
import tomllib
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
dataset, method = sys.argv[1], sys.argv[2]
os.environ["RECRAG_RUN_NAMESPACE"] = f"final500/{dataset}"
os.environ.setdefault("RECRAG_BUILD_SERVICE_TIER", "flex")
if dataset == "hotpotqa":
    os.environ["RECRAG_HOTPOTQA_ROWS"] = str(
        ROOT / "runs/final500/hotpotqa/questions_rows.json"
    )
elif dataset == "natural_questions":
    os.environ["RECRAG_NQ_CORPUS_ROWS"] = "0-499"
    os.environ["RECRAG_NQ_RECORDS"] = str(
        ROOT / "runs/final500/natural_questions/records.json"
    )
else:
    raise SystemExit("dataset must be hotpotqa or natural_questions")

from dotenv import load_dotenv

import benchmark
from cost import metering_builds, unmetered
from dataset import (
    load_hotpotqa_corpus,
    load_natural_questions_corpus,
    natural_questions_corpus_rows,
)

load_dotenv(ROOT / ".env", override=True)
settings = benchmark.Settings.model_validate(
    tomllib.loads((ROOT / "config.toml").read_text())
)
pages = (
    load_hotpotqa_corpus()
    if dataset == "hotpotqa"
    else load_natural_questions_corpus(natural_questions_corpus_rows())
)
documents = benchmark.split_pages(pages)
corpus_key = sha256("\n\n".join(documents).encode()).hexdigest()
run_root = ROOT / "runs" / "final500" / dataset
index_dir = (
    run_root
    / method
    / "indexes"
    / benchmark.index_key(method, dataset, corpus_key, settings)
)
out = run_root / method / "build"
out.mkdir(parents=True, exist_ok=True)

seeded = None
if method == "graphrag" and dataset == "natural_questions":
    # Reuse the NQ 100 pilot's graph-extraction results. The cache key is the
    # hash of the construction identity and the full extraction prompt, which
    # holds the exact chunk text, so only identical requests can hit.
    database = f"recrag-benchmark-{index_dir.name[:24]}"
    target = run_root / "graphrag" / "construction-cache" / f"{database}.sqlite"
    if not target.exists():
        sources = sorted(
            (ROOT / "runs/nq100/graphrag/construction-cache").glob("*.sqlite")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS chat_generations ("
                "key TEXT PRIMARY KEY, result TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            for source in sources:
                # ATTACH reads through SQLite, so rows still in a source's
                # write-ahead log are included.
                db.execute("ATTACH DATABASE ? AS src", [str(source)])
                db.execute(
                    "INSERT OR IGNORE INTO chat_generations "
                    "SELECT key, result FROM src.chat_generations"
                )
                db.execute(
                    "INSERT OR IGNORE INTO embeddings "
                    "SELECT key, vector FROM src.embeddings"
                )
                db.commit()
                db.execute("DETACH DATABASE src")
            seeded = {
                "sources": len(sources),
                "generations": db.execute(
                    "SELECT count(*) FROM chat_generations"
                ).fetchone()[0],
                "embeddings": db.execute("SELECT count(*) FROM embeddings").fetchone()[
                    0
                ],
            }

started_at = datetime.now(UTC)
started = perf_counter()
# Build calls are not retried, so one transient server error (seen: HTTP 520)
# stops a build. Each builder caches every finished call, so a restart resumes
# where it stopped; the same meter covers every attempt.
ATTEMPTS = 20
errors: list[str] = []
purged: list[dict] = []


def purge_map_properties(index_dir: Path) -> None:
    """Drop cached graph extractions with a map-valued property.

    Neo4j stores only primitive property values; one Natural Questions reply
    gave "coordinates": {"latitude": ..., "longitude": ...}, so writing the
    graph fails on every restart. Deleting that cached reply makes the next
    attempt ask the model again (a retry of that one call); prompts and the
    build are unchanged. Logged in purged_construction_rows.*.json.
    """
    database = f"recrag-benchmark-{index_dir.name[:24]}"
    path = run_root / "graphrag" / "construction-cache" / f"{database}.sqlite"
    rows = []
    with sqlite3.connect(path) as db:
        for key, result in db.execute(
            "SELECT key, result FROM chat_generations"
        ).fetchall():
            generation = json.loads(json.loads(result)[0])
            body = generation.get("text") or (generation.get("message") or {}).get(
                "content"
            )
            try:
                graph = json.loads(str(body))
            except ValueError:
                continue
            items = (
                graph.get("nodes", []) + graph.get("relationships", [])
                if isinstance(graph, dict)
                else []
            )
            if any(
                isinstance(p.get("value"), dict)
                for x in items
                for p in x.get("properties", [])
            ):
                rows.append({"key": key, "reply": str(body)[:600]})
                db.execute("DELETE FROM chat_generations WHERE key = ?", [key])
        db.commit()
    (
        out / f"purged_construction_rows.{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    ).write_text(json.dumps(rows, indent=1))


def purge_rejected_openie(index_dir: Path) -> None:
    """Drop cached HippoRAG OpenIE replies its own parser rejects.

    HippoRAG caches every reply before parsing it. A reply in the wrong shape
    (seen: {"entities": [...]} instead of {"named_entities": [...]}) is
    rejected, the build stops, and every restart replays the same cached
    reply. Deleting such a row makes the next attempt ask the model again, the
    same as retrying that one call; prompts and parsing are unchanged. Every
    dropped row is logged in build.json ("purged_cache_rows").
    """
    for path in index_dir.glob("responses_cache/*.sqlite"):
        with sqlite3.connect(path) as db:
            for key, message in db.execute("SELECT key, message FROM cache").fetchall():
                try:
                    value = json.loads(message)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict) and not (
                    {"named_entities", "triples"} & set(value)
                ):
                    db.execute("DELETE FROM cache WHERE key = ?", [key])
                    purged.append({"key": key, "reply": message[:300]})
            db.commit()
    (out / "purged_cache_rows.json").write_text(json.dumps(purged, indent=1))


with metering_builds(settings.pricing) as build_cost:
    for attempt in range(ATTEMPTS):
        try:
            with (
                benchmark.method_setup(method)(
                    settings, index_dir, dataset
                ) as pipeline,
                unmetered(),
            ):
                benchmark.search_method(
                    benchmark.WARMUP_QUESTION,
                    method=method,
                    pipeline=pipeline,
                    documents=documents,
                    pages=pages,
                    corpus_key=corpus_key,
                    indexed_corpora=set(),
                    index_dir=index_dir,
                    top_k=settings.retrieval.top_k,
                )
            break
        except Exception as exc:  # noqa: BLE001 - restart from the caches
            errors.append(f"{datetime.now(UTC).isoformat()} {exc!r}"[:500])
            (out / "build_errors.log").write_text("\n".join(errors) + "\n")
            (out / "build_cost.partial.json").write_text(build_cost.model_dump_json())
            print("BUILD ATTEMPT FAILED", attempt + 1, repr(exc)[:300], flush=True)
            if attempt + 1 == ATTEMPTS:
                raise
            if (
                method == "graphrag"
                and "Property values can only be of primitive types" in str(exc)
            ):
                purge_map_properties(index_dir)
            if method == "hipporag":
                from openie_purge import chunk_ids_in, purge

                with benchmark.method_setup(method)(
                    settings, index_dir, dataset
                ) as cleanup:
                    purged.extend(
                        purge(cleanup, documents, chunk_ids_in(str(exc)) or None)
                    )
                (
                    out / f"purged_openie_rows.{started_at:%Y%m%dT%H%M%S}.json"
                ).write_text(json.dumps(purged, indent=1))
            time.sleep(60)
seconds = perf_counter() - started
(out / "build_cost.json").write_text(
    benchmark.BuildCostReport(
        cost=build_cost,
        list_usd=build_cost.list_usd(settings.pricing),
        billed_usd=build_cost.estimated_usd,
    ).model_dump_json(indent=2)
)
tiers: dict[str, int] = {}
for call in build_cost.calls:
    tiers[call.service_tier] = tiers.get(call.service_tier, 0) + 1
(out / "build.json").write_text(
    json.dumps(
        {
            "dataset": dataset,
            "method": method,
            "pages": len(pages),
            "chunks": len(documents),
            "corpus_sha256": corpus_key,
            "index_dir": str(index_dir),
            "started_at": started_at.isoformat(),
            "seconds": seconds,
            "service_tier_env": os.environ.get("RECRAG_BUILD_SERVICE_TIER"),
            "calls_by_tier": tiers,
            "failed_calls": len(build_cost.failed_calls),
            "restarts": errors,
            "purged_cache_rows": len(purged),
            "missing_usage_calls": build_cost.missing_usage_calls,
            "billed_usd": build_cost.estimated_usd,
            "list_usd": build_cost.list_usd(settings.pricing),
            "seeded_cache": seeded,
        },
        indent=1,
    )
)
print("BUILT", dataset, method, round(seconds), "s", "billed", build_cost.estimated_usd)
