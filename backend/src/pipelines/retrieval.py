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
    get_vector_store_paths,
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
    """Pipeline for retrieving and generating responses.

    Supports dependency injection for flexible composition.
    """

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

        index_path, metadata_path = get_vector_store_paths(
            config, config_path, embedder.model
        )
        vector_store = VectorStore(
            dimension=embedder.dimension,
            index_path=index_path,
            metadata_path=metadata_path,
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
        """Retrieve relevant documents for a query."""
        k = top_k or self.top_k
        logger.info(f"Embedding query: {query[:50]}...")

        query_embedding = self.embedder.embed(query)
        _, results = self.vector_store.search(query_embedding, k=k)

        logger.info(f"Found {len(results)} results")
        return results

    def generate(
        self,
        query: str,
        context: Optional[list[RetrievalResult]] = None,
    ) -> str:
        """Generate a response using retrieved context."""
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

        logger.info("Generating response...")
        return self.llm.generate(prompt)

    def query(self, query: str) -> dict[str, Any]:
        """Execute a full RAG query: retrieve and generate."""
        context = self.retrieve(query)
        response = self.generate(query, context)

        return {"response": response, "context": context}


def get_retrieval_pipeline(
    config_path: Path = Path("config.toml"),
) -> RetrievalPipeline:
    """Create a retrieval pipeline from config.

    Args:
        config_path: Path to configuration file.

    Returns:
        RetrievalPipeline instance.
    """
    config = load_config(config_path)
    return RetrievalPipeline.from_config(config, config_path)
