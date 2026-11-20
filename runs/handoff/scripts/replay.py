"""Hand-off replays (runs/handoff/SPEC.md): same saved agent searches, new hand-off rule.

Usage (repo root):
  PYTHONPATH=. .venv/bin/python runs/handoff/scripts/replay.py build
  PYTHONPATH=. .venv/bin/python runs/handoff/scripts/replay.py answer VARIANT REP

build: reads the saved agent runs (A3 = runs/final500 Agentic Hybrid RAG, A1 =
runs/nograph), rebuilds every tool result from the saved trace, and writes one
retrieval.jsonl per variant and repeat under runs/handoff/runs/<variant>/r<rep>/.
No model call is made except query embeddings (text-embedding-3-small), needed
to score passages for the old top-10 rule. Before writing anything it checks
that the agent's own rule ("picks"), rebuilt from the trace, gives exactly the
saved evidence for every row.

answer: runs benchmark.answer_saved on one retrieval.jsonl (same answer model,
prompt and settings as the source run), writing results.jsonl next to it.
"""

import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs/handoff/runs"
IDS = set(json.loads((ROOT / "runs/nograph/questions.json").read_text())["indices"])
SOURCES = {
    "A3": json.loads((ROOT / "runs/final500/run_registry.json").read_text())["multihop_rag"]["graphrag"],
    "A1": json.loads((ROOT / "runs/nograph/run_registry.json").read_text()),
}
# variant: (source, rule)
VARIANTS = {
    "A2_graph_top10": ("A3", "top10"),
    "A0_vector_top10": ("A1", "top10"),
    "A3_picks_no_graph_rows": ("A3", "picks_vector_only"),
}
LINE = re.compile(r"^\[(E[0-9a-f]{6})\] (.*)$")


def rows_of(source: str, rep: str) -> tuple[list[dict], Path]:
    dirs = [Path(d) for d in SOURCES[source][rep]]
    rows = [
        json.loads(line)
        for d in dirs
        if (d / "retrieval.jsonl").exists()
        for line in open(d / "retrieval.jsonl")
    ]
    rows = [r for r in rows if int(r["record_id"].rsplit("-", 1)[1]) in IDS]
    assert len(rows) == 250 and len({r["record_id"] for r in rows}) == 250, (source, rep, len(rows))
    return rows, dirs[0]


def tool_results(row: dict) -> list[tuple[int, str, str, str]]:
    """(step, tool, id, content) for every result line, in arrival order."""
    out = []
    for t in row["retrieval"]["trace"]:
        if t["tool"] not in ("vector_search", "cypher_search") or t["status"] != "success":
            continue
        for line in t["output"].split("\n"):
            m = LINE.match(line)
            if m:
                out.append((t["step"], t["tool"], m.group(1), m.group(2)))
    return out


def main_build() -> None:
    import graphrag_retrieval as g

    rag = g.GraphRAG.from_config(dataset="multihop_rag")
    (ROOT / "runs/handoff/cache").mkdir(parents=True, exist_ok=True)
    rag._embedder.store = g.SQLiteEmbeddingCache(ROOT / "runs/handoff/cache/query_embeddings.sqlite")
    chunk_vec: dict[tuple[str, str], list[float]] = {}

    def score(db: str, query: str, text: str) -> float:
        # Neo4j's cosine vector index reports (1 + cos) / 2.
        if (db, text) not in chunk_vec:
            recs = rag._driver.execute_query(
                "MATCH (c:Chunk) WHERE c.text = $t RETURN c.embedding AS e LIMIT 1",
                t=text, database_=db,
            ).records
            chunk_vec[(db, text)] = recs[0]["e"]
        q = rag._embedder.embed_query(query)
        c = chunk_vec[(db, text)]
        cos = sum(a * b for a, b in zip(q, c)) / (math.hypot(*q) * math.hypot(*c))
        return (1 + cos) / 2

    def evidence_lists(row: dict) -> dict[str, list[tuple[str, str, float]]]:
        """Rebuild the agent's pooled results; return each rule's 10 items (id, content, score)."""
        db = "recrag-benchmark-" + Path(row["index_dir"]).name[:24]
        queries = {t["step"]: t["args"].get("query_text", "") for t in row["retrieval"]["trace"]}
        passages: dict[str, tuple[float, int, str, str]] = {}
        seen, graph_rows = set(), []
        for step, tool, eid, content in tool_results(row):
            key = g._evidence_key(content)
            assert g.evidence_id(content) == eid
            if tool == "vector_search":
                s = score(db, queries[step], g.parse_vector_evidence(content).text)
                found = passages.get(key)
                if found is None:
                    passages[key] = (s, len(passages), eid, content)
                elif s > found[0]:
                    passages[key] = (s, found[1], eid, content)
            elif key not in seen:
                seen.add(key)
                graph_rows.append((eid, content, g.NO_SCORE))
        ranked = [(e, c, s) for s, _, e, c in sorted(passages.values(), key=lambda x: (-x[0], x[1]))]
        fallback = ranked + graph_rows
        by_id = {e: (e, c, s) for e, c, s in fallback}
        reply = row["retrieval"]["trace"][-1]["output"]
        picked = []
        for eid in g.EVIDENCE_ID.findall(reply):
            if eid in by_id and by_id[eid] not in picked:
                picked.append(by_id[eid])

        def fill(first, pool):
            kept = first[:10]
            return kept + [x for x in pool if x not in kept][: 10 - len(kept)]

        return {
            "picks": fill(picked, fallback),
            "top10": fallback[:10],
            "picks_vector_only": fill([p for p in picked if p[2] != g.NO_SCORE], ranked),
        }

    stats: Counter = Counter()
    for source in ("A3", "A1"):
        for rep in ("1", "2", "3"):
            rows, settings_dir = rows_of(source, rep)
            built = []
            for row in rows:
                lists = evidence_lists(row)
                saved = row["retrieval"]["evidence"]
                again = [c for _, c, _ in lists["picks"]]
                if again != saved:
                    stats[f"{source} picks mismatch"] += 1
                    raise SystemExit(f"{source} r{rep} {row['record_id']}: rebuilt picks differ from saved evidence")
                # Saved scores of kept vector passages, vs ours, as a check on the re-embedding.
                for (_, _, s), old in zip(lists["picks"], row["retrieval"]["scores"]):
                    if old != g.NO_SCORE:
                        stats["score checks"] += 1
                        stats["max score diff x1e6"] = max(stats["max score diff x1e6"], round(abs(s - old) * 1e6))
                built.append((row, lists))
            for variant, (src, rule) in VARIANTS.items():
                if src != source:
                    continue
                d = OUT / variant / f"r{rep}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(settings_dir / "settings.json", d / "settings.json")
                with open(d / "retrieval.jsonl", "w") as f:
                    for row, lists in built:
                        items = lists[rule]
                        new = json.loads(json.dumps(row))
                        new["retrieval"]["evidence"] = [c for _, c, _ in items]
                        new["retrieval"]["passages"] = [g.answer_text(c) for _, c, _ in items]
                        new["retrieval"]["scores"] = [s for _, _, s in items]
                        new["retrieval"]["trace"].append(
                            {"step": len(row["retrieval"]["trace"]), "tool": "handoff_replay",
                             "args": {"variant": variant, "rule": rule, "source_run": source},
                             "status": "success", "output": "", "results": len(items)}
                        )
                        changed = new["retrieval"]["evidence"] != row["retrieval"]["evidence"]
                        stats[f"{variant} r{rep} evidence changed"] += changed
                        stats[f"{variant} r{rep} graph rows sent"] += sum(s == g.NO_SCORE for _, _, s in items)
                        f.write(json.dumps(new, ensure_ascii=False) + "\n")
            print(source, rep, "ok", flush=True)
    rag.close()
    (ROOT / "runs/handoff/build_check.json").write_text(json.dumps(dict(stats), indent=1) + "\n")
    print(json.dumps(dict(stats), indent=1))


def main_answer(variant: str, rep: str) -> None:
    import benchmark

    path = OUT / variant / f"r{rep}" / "retrieval.jsonl"
    print("RESULTS", benchmark.answer_saved(path), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        main_build()
    else:
        main_answer(sys.argv[2], sys.argv[3])
