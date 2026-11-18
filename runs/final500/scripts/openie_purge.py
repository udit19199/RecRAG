"""Delete cached HippoRAG OpenIE replies that HippoRAG's own parser rejects.

HippoRAG caches every model reply before parsing it. A reply in the wrong
shape (seen: {"entities": ...} for NER, a reply without "triples" for triple
extraction) is rejected, the build stops, and every restart replays the same
cached reply. This replays HippoRAG's own ner() and triple_extraction() for
the given chunks, reading from the cache only; when a cached reply is
rejected, its exact cache row (the key HippoRAG itself looked up) is deleted,
so the next build attempt asks the model again, the same as retrying that one
call. Prompts, parsing and the build are unchanged.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hipporag import HippoRAG
from hipporag.llm import openai_gpt
from hipporag.utils.misc_utils import compute_mdhash_id

_seen = threading.local()


class _Cursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=()):
        if sql.lstrip().startswith("SELECT message, metadata FROM cache"):
            _seen.key = params[0]
        self._cursor.execute(sql, params)
        return self

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _Connection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self):
        return _Cursor(self._connection.cursor())

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, *exc):
        return self._connection.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _SQLite:
    def __getattr__(self, name):
        return getattr(sqlite3, name)

    @staticmethod
    def connect(*args, **kwargs):
        return _Connection(sqlite3.connect(*args, **kwargs))


def purge(pipeline, documents: list[str], chunk_ids: set[str] | None) -> list[dict]:
    """Return one entry per deleted cache row."""
    original = openai_gpt.sqlite3
    openai_gpt.sqlite3 = _SQLite()  # type: ignore[assignment]
    deleted: list[dict] = []
    try:
        rag = HippoRAG(
            global_config=pipeline.config,
            extraction_llm=pipeline.extraction_llm,
            qa_llm=pipeline.llm,
            embedding_model=pipeline.embeddings,
            index_identity="recrag-paper-chunks256-overlap20-v1",
        )
        cache_file = pipeline.extraction_llm.cache_file_name
        for passage in documents:
            chunk = compute_mdhash_id(passage, prefix="chunk-")
            if chunk_ids is not None and chunk not in chunk_ids:
                continue
            for stage in ("ner", "triples"):
                _seen.key = None
                if stage == "ner":
                    result = rag.openie.ner(chunk, passage)
                    entities = result.unique_entities
                else:
                    result = rag.openie.triple_extraction(chunk, passage, entities)
                error = result.metadata.get("error")
                if error and result.metadata.get("cache_hit") and _seen.key:
                    with sqlite3.connect(cache_file) as db:
                        db.execute("DELETE FROM cache WHERE key = ?", [_seen.key])
                    deleted.append(
                        {
                            "chunk": chunk,
                            "stage": stage,
                            "key": _seen.key,
                            "error": error[:200],
                            "reply": (result.response or "")[:300],
                        }
                    )
                    break
                if error:
                    break
        rag.close() if hasattr(rag, "close") else None
    finally:
        openai_gpt.sqlite3 = original  # type: ignore[assignment]
    return deleted


def chunk_ids_in(message: str) -> set[str]:
    import re

    return set(re.findall(r"chunk-[0-9a-f]{32}", message))


__all__ = ["purge", "chunk_ids_in", "Path"]
