"""Dense vector retrieval used by the Normal RAG baseline."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from openai import OpenAI
from pydantic import PrivateAttr

from benchmark import RetrievedContext, Settings, Usage
from cost import MeteredClient, indexing


class LlamaEmbedding(BaseEmbedding):
    _adapter: OpenAI = PrivateAttr()
    _dimensions: int = PrivateAttr()

    def __init__(self, adapter: OpenAI, model_name: str, dimensions: int) -> None:
        super().__init__(model_name=model_name)
        self._adapter = adapter
        self._dimensions = dimensions

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = self._adapter.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=self._dimensions,
            encoding_format="float",
        )
        rows = sorted(response.data, key=lambda item: item.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise RuntimeError("Embedding response did not match the input documents.")
        if any(len(row.embedding) != self._dimensions for row in rows):
            raise RuntimeError("Embedding response had unexpected dimensions.")
        return [row.embedding for row in rows]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)


class ResponsesLLM:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.usage: list[Usage] = []

    def infer(self, messages):
        response = self.client.responses.create(
            model=self.settings.llm.model,
            input=messages,
            reasoning={"effort": self.settings.llm.reasoning_effort},
            store=False,
        )
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(
                f"Normal RAG generation did not complete: {response.status}"
            )
        if response.usage is None:
            raise RuntimeError("Normal RAG generation returned no token usage.")
        self.usage.append(
            Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            )
        )
        return [response.output_text]


@contextmanager
def setup(settings: Settings, index_dir: Path, dataset: str):
    # Normal RAG owns its clients and does not depend on HippoRAG adapters.
    with (
        OpenAI(
            timeout=settings.llm.timeout,
            max_retries=0,
            http_client=MeteredClient(timeout=settings.llm.timeout),
        ) as llm_client,
        OpenAI(
            timeout=settings.embedding.timeout,
            max_retries=0,
            http_client=MeteredClient(timeout=settings.embedding.timeout),
        ) as embedding_client,
    ):
        index_dir.mkdir(parents=True, exist_ok=True)
        yield SimpleNamespace(
            llm=ResponsesLLM(llm_client, settings),
            embeddings=LlamaEmbedding(
                embedding_client,
                settings.embedding.model,
                settings.embedding.dimensions,
            ),
        )


def retrieve(
    *,
    documents: list[str],
    question: str,
    index_dir: Path,
    embeddings: LlamaEmbedding,
    top_k: int,
) -> RetrievedContext:
    started = perf_counter()
    embedding = embeddings
    if (index_dir / "index_store.json").exists():
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(index_dir)),
            embed_model=embedding,
        )
        construction_seconds = 0.0
    else:
        with indexing():
            vector_index = VectorStoreIndex(
                [TextNode(text=document) for document in documents],
                embed_model=embedding,
            )
        vector_index.storage_context.persist(persist_dir=str(index_dir))
        construction_seconds = perf_counter() - started

    started = perf_counter()
    hits = vector_index.as_retriever(similarity_top_k=top_k).retrieve(question)
    return RetrievedContext(
        passages=[hit.node.get_content() for hit in hits],
        scores=[float(hit.score or 0.0) for hit in hits],
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from benchmark import main

    main(default_method="rag")
