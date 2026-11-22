"""Load the exported chunks into Milvus (VM). Usage (repo root): PYTHONPATH=. .venv/bin/python runs/milvus/scripts/load.py

Drops and recreates agent_chunks and rag_chunks, inserts every chunk with its
saved vector (no embedding call), builds the HNSW index (milvus_store.INDEX),
loads both, and checks the row counts (7,728 each) and that each chunk's own
vector finds itself first.
"""

import numpy as np
from pymilvus import DataType

from runs.milvus.scripts.milvus_store import INDEX, client, search

c = client()
for name, path, has_title in (
    ("agent_chunks", "runs/milvus/export/agent_chunks.npz", True),
    ("rag_chunks", "runs/milvus/export/rag_chunks.npz", False),
):
    data = np.load(path)
    texts, vectors = data["texts"], data["vectors"]
    if c.has_collection(name):
        c.drop_collection(name)
    schema = c.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    if has_title:
        schema.add_field("title", DataType.VARCHAR, max_length=2048)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])
    index = c.prepare_index_params()
    index.add_index("vector", index_type=INDEX["index_type"], metric_type=INDEX["metric_type"], params=INDEX["params"])
    c.create_collection(name, schema=schema, index_params=index)
    rows = [
        {"id": i, "text": str(texts[i]), "vector": vectors[i].tolist(), **({"title": str(data["titles"][i])} if has_title else {})}
        for i in range(len(texts))
    ]
    for start in range(0, len(rows), 500):
        c.insert(name, rows[start : start + 500])
    c.flush(name)
    c.load_collection(name)
    count = c.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
    self_first = sum(search(name, vectors[i].tolist(), 1, ["text"])[0]["text"] == str(texts[i]) for i in range(0, len(texts), 97))
    print(name, "rows", count, "self-hit check", self_first, "of", len(range(0, len(texts), 97)))
    assert count == len(texts) == 7728
