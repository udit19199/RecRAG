"""Dense vector retrieval used by the Normal RAG baseline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from pydantic import PrivateAttr

from . import RetrievedContext

if TYPE_CHECKING:
    from .runner import ConfiguredEmbeddings


class LlamaEmbedding(BaseEmbedding):
    _adapter: ConfiguredEmbeddings = PrivateAttr()

    def __init__(self, adapter: ConfiguredEmbeddings, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self._adapter = adapter

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._adapter.batch_encode([text])[0].tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._adapter.batch_encode(texts).tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)


def retrieve(
    *,
    documents: list[str],
    question: str,
    index_dir: Path,
    embeddings: ConfiguredEmbeddings,
    embedding_model: str,
    top_k: int,
) -> RetrievedContext:
    started = perf_counter()
    embedding = LlamaEmbedding(embeddings, embedding_model)
    if (index_dir / "index_store.json").exists():
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir)),
            embed_model=embedding,
        )
        construction_seconds = 0.0
    else:
        vector_index = VectorStoreIndex(
            [TextNode(text=document) for document in documents],
            embed_model=embedding,
        )
        vector_index.storage_context.persist(persist_dir=str(index_dir))
        construction_seconds = perf_counter() - started

    started = perf_counter()
    hits = vector_index.as_retriever(similarity_top_k=top_k).retrieve(question)
    return RetrievedContext(
        passages=[hit.node.text for hit in hits],
        scores=[float(hit.score or 0.0) for hit in hits],
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from .runner import main

    main(default_method="rag")
