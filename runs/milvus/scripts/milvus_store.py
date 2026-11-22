"""Milvus helpers for runs/milvus (self-hosted Milvus 2.5.10 standalone on the Instabox VM).

Collections (loaded by load.py from runs/milvus/export/*.npz):
  agent_chunks: id, title, text, vector (the Agentic Hybrid RAG graph's chunk embeddings)
  rag_chunks:   id, text, vector (Normal RAG's llama-index node embeddings)
Index: HNSW, metric COSINE, M=16, efConstruction=200; search ef=128.
Milvus COSINE returns the cosine similarity. search() returns it unchanged;
callers that mirror Neo4j's cosine index convert it to (1 + cos) / 2.
"""

import os

from pymilvus import MilvusClient

URI = os.environ.get("RECRAG_MILVUS_URI", "http://127.0.0.1:19530")
INDEX = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}}
SEARCH = {"metric_type": "COSINE", "params": {"ef": 128}}
_client: MilvusClient | None = None


def client() -> MilvusClient:
    global _client
    if _client is None:
        _client = MilvusClient(uri=URI)
    return _client


def search(collection: str, vector: list[float], k: int, fields: list[str]) -> list[dict]:
    """Top k hits, best first: each hit's fields plus "score" (cosine similarity)."""
    hits = client().search(
        collection, data=[vector], limit=k, output_fields=fields, search_params=SEARCH
    )[0]
    return [{**{f: h["entity"][f] for f in fields}, "score": float(h["distance"])} for h in hits]
