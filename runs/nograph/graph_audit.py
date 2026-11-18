"""Graph audit (plan step "Graph audit, 4 Neo4j counts"), read-only.

Usage (repo root): PYTHONPATH=. .venv/bin/python runs/nograph/graph_audit.py
Runs Bhanu's four queries (18 Nov 2026) on the MultiHop-RAG graph used by the
latest run and writes runs/nograph/graph_audit.json.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
DB = "recrag-benchmark-e5315c2b5c9f57e2595cc51c"
QUERIES = {
    "entities_with_list_name": "MATCH (n:__Entity__) WHERE NOT n.name IS :: STRING RETURN count(n) AS n",
    "publisher_nodes": "MATCH (p:Publisher) RETURN count(p) AS n",
    "duplicate_name_groups_case_or_space": (
        "MATCH (n:__Entity__) WHERE n.name IS :: STRING "
        "WITH toLower(trim(n.name)) AS key, count(*) AS c WHERE c > 1 "
        "RETURN count(key) AS n"
    ),
    "chunks_with_no_entity": "MATCH (c:Chunk) WHERE NOT EXISTS { (:__Entity__)-[:FROM_CHUNK]->(c) } RETURN count(c) AS n",
    # Denominators, so the counts can be read as shares.
    "entities_total": "MATCH (n:__Entity__) RETURN count(n) AS n",
    "chunks_total": "MATCH (c:Chunk) RETURN count(c) AS n",
    "distinct_publisher_names": "MATCH (p:Publisher) RETURN count(DISTINCT toLower(trim(toString(p.name)))) AS n",
    "entities_in_duplicate_groups": (
        "MATCH (n:__Entity__) WHERE n.name IS :: STRING "
        "WITH toLower(trim(n.name)) AS key, count(*) AS c WHERE c > 1 "
        "RETURN sum(c) AS n"
    ),
}
auth = os.environ["NEO4J_AUTH"]
user, password = auth.split("/", 1) if "/" in auth else ["neo4j", auth]
with GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(user, password)) as driver:
    out = {"database": DB, "queries": QUERIES, "counts": {}}
    for name, q in QUERIES.items():
        out["counts"][name] = driver.execute_query(q, database_=DB).records[0]["n"]
        print(f"{name}: {out['counts'][name]}")
Path("runs/nograph/graph_audit.json").write_text(json.dumps(out, indent=1) + "\n")
