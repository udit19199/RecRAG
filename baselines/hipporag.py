"""Native graph retrieval used by the HippoRAG baseline."""

from __future__ import annotations

from time import perf_counter

from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm.base import BaseLLM
from hipporag.utils.config_utils import BaseConfig

from . import RetrievedContext


def retrieve(
    *,
    config: BaseConfig,
    documents: list[str],
    corpus_key: str,
    indexed_corpora: set[str],
    llm: BaseLLM,
    embeddings: OpenAIEmbeddingModel,
    question: str,
    top_k: int,
) -> RetrievedContext:
    started = perf_counter()
    with HippoRAG(
        global_config=config,
        extraction_llm=llm,
        qa_llm=llm,
        embedding_model=embeddings,
        index_identity="recrag-paper-chunks256-overlap20-v1",
    ) as rag:
        if corpus_key not in indexed_corpora:
            rag.index(docs=documents)
            indexed_corpora.add(corpus_key)
            construction_seconds = perf_counter() - started
        else:
            construction_seconds = 0.0
        started = perf_counter()
        solution = rag.retrieve(queries=[question], num_to_retrieve=top_k)[0]
    return RetrievedContext(
        passages=solution.docs,
        scores=solution.doc_scores.tolist(),
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from .runner import main

    main(default_method="hipporag")
