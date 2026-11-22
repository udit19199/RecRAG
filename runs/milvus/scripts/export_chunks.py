"""Export the MultiHop-RAG chunks and their saved embeddings for Milvus (runs/milvus/SPEC.md).

Usage (repo root, Mac, light): PYTHONPATH=. .venv/bin/python runs/milvus/scripts/export_chunks.py
No model call. Writes runs/milvus/export/{agent_chunks,rag_chunks}.npz:
  agent_chunks: the 7,728 chunks of the Agentic Hybrid RAG graph (benchmark.split_pages per
    page, page title kept), vectors from the graph build's embedding cache
    (runs/graphrag/construction-cache/recrag-benchmark-e5315c2b5c9f57e2595cc51c.sqlite).
  rag_chunks: Normal RAG's 7,728 llama-index nodes, vectors from its saved index.
Every chunk must be found; the script stops otherwise.
"""
import json, sqlite3
from array import array
from hashlib import sha256
from pathlib import Path
import numpy as np
import benchmark
from dataset import get_source

OUT = Path("runs/milvus/export"); OUT.mkdir(parents=True, exist_ok=True)
pages = get_source("multihop_rag").load_record(306).pages
chunks = [(p.title or "", c) for p in pages for c in benchmark.split_pages([p])]
db = sqlite3.connect("runs/graphrag/construction-cache/recrag-benchmark-e5315c2b5c9f57e2595cc51c.sqlite")
vec = []
for _, c in chunks:
    k = sha256(f"text-embedding-3-small\x001536\x00{c}".encode()).hexdigest()
    (blob,) = db.execute("select vector from embeddings where key=?", [k]).fetchone()
    vec.append(array("d", blob).tolist())
np.savez_compressed(OUT / "agent_chunks.npz", titles=np.array([t for t, _ in chunks]),
                    texts=np.array([c for _, c in chunks]), vectors=np.array(vec, dtype=np.float32))
I = Path("runs/rag/indexes/07062033a2e929ec10405ef3a38d541e51b907cbb7041de545f72ab1e7b71799")
emb = json.loads((I / "default__vector_store.json").read_text())["embedding_dict"]
docs = json.loads((I / "docstore.json").read_text())["docstore/data"]
ids = list(emb)
assert set(ids) == set(docs) and len(ids) == 7728
np.savez_compressed(OUT / "rag_chunks.npz", node_ids=np.array(ids),
                    texts=np.array([docs[i]["__data__"]["text"] for i in ids]),
                    vectors=np.array([emb[i] for i in ids], dtype=np.float32))
print("agent_chunks", len(chunks), "rag_chunks", len(ids))
