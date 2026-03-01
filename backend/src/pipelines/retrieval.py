import logging
from pathlib import Path
from typing import Any, Optional

import tiktoken
from adapters import BaseEmbedder, BaseLLM
from config import get_config_value, load_config
from models.chunk import RetrievalResult
from stores import BaseVectorStore, VectorStore
from .base import (
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_TOP_K,
    create_embedder_from_config,
    create_llm_from_config,
    get_collection_name,
    get_milvus_uri,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 4096
ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def get_tokenizer(model: str) -> tiktoken.Encoding:
    if model not in ENCODING_CACHE:
        try:
            ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return ENCODING_CACHE[model]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoder = get_tokenizer(model)
    return len(encoder.encode(text))


class RetrievalPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        top_k: int = DEFAULT_TOP_K,
        context_template: str = DEFAULT_CONTEXT_TEMPLATE,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        config: dict[str, Any] | None = None,
        config_path: Path | None = None,
    ):
        self.embedder = embedder
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.context_template = context_template
        self.max_context_tokens = max_context_tokens
        self.config = config or {}
        self.config_path = config_path

    @classmethod
    def from_config(
        cls, config: dict[str, Any], config_path: Path
    ) -> "RetrievalPipeline":
        """Create pipeline from configuration dictionary."""
        embedder = create_embedder_from_config(config)
        llm = create_llm_from_config(config)

        collection_name = get_collection_name(config, embedder.model)
        uri = get_milvus_uri(config, config_path)
        vector_store = VectorStore(
            dimension=embedder.dimension,
            collection_name=collection_name,
            uri=uri,
            metric_type=config.get("storage", {}).get("metric_type", "L2"),
        )

        top_k = get_config_value(config, "retrieval.top_k", DEFAULT_TOP_K)
        max_context_tokens = get_config_value(
            config, "retrieval.max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS
        )
        context_template = config.get("retrieval", {}).get(
            "context_template", DEFAULT_CONTEXT_TEMPLATE
        )

        return cls(
            embedder=embedder,
            llm=llm,
            vector_store=vector_store,
            top_k=top_k,
            context_template=context_template,
            max_context_tokens=max_context_tokens,
            config=config,
            config_path=config_path,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        query_embedding = self.embedder.embed(query)
        _, results = self.vector_store.search(query_embedding, k=k)
        return results

    def generate(
        self,
        query: str,
        context: Optional[list[RetrievalResult]] = None,
    ) -> str:
        context = context or self.retrieve(query)

        model = getattr(self.llm, "model", "gpt-4")
        template_overhead = count_tokens(
            self.context_template.format(context="", question=query), model
        )
        available_tokens = self.max_context_tokens - template_overhead

        context_text = ""
        current_tokens = 0
        truncated = False

        for doc in context:
            doc_text = doc.text
            doc_tokens = count_tokens(doc_text, model)

            if current_tokens + doc_tokens <= available_tokens:
                if context_text:
                    context_text += "\n\n"
                context_text += doc_text
                current_tokens += doc_tokens
            else:
                truncated = True
                break

        if truncated:
            logger.warning(
                f"Context truncated to {current_tokens} tokens (limit: {self.max_context_tokens})"
            )

        prompt = self.context_template.format(
            context=context_text,
            question=query,
        )

        return self.llm.generate(prompt)

    def query(self, query: str) -> dict[str, Any]:
        context = self.retrieve(query)
        response = self.generate(query, context)

        return {"response": response, "context": context}


def get_retrieval_pipeline(
    config_path: Path = Path("config.toml"),
) -> RetrievalPipeline:
    config = load_config(config_path)
    return RetrievalPipeline.from_config(config, config_path)
