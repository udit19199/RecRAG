import logging
from pathlib import Path
from typing import Any, Optional

import tiktoken
from adapters import BaseEmbedder, BaseLLM
from config import get_config_value, load_config
from models.chunk import RetrievalResult
from stores import VectorStore
from .base import (
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_TOP_K,
    create_embedder_from_config,
    create_llm_from_config,
    create_vector_store_from_config,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 4096
_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_tokenizer(model: str) -> tiktoken.Encoding:
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODING_CACHE[model]


def _count_tokens(text: str, model: str = "gpt-4") -> int:
    return len(_get_tokenizer(model).encode(text))


class RetrievalPipeline:
    def __init__(
        self,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        vector_store: VectorStore,
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
        cls, config: dict[str, Any], config_path: Path, vision_model: str | None = None
    ) -> "RetrievalPipeline":
        embedder = create_embedder_from_config(config)
        llm = create_llm_from_config(config)

        vector_store = create_vector_store_from_config(
            config, config_path, embedder, vision_model
        )

        return cls(
            embedder=embedder,
            llm=llm,
            vector_store=vector_store,
            top_k=get_config_value(config, "retrieval.top_k", DEFAULT_TOP_K),
            context_template=config.get("retrieval", {}).get(
                "context_template", DEFAULT_CONTEXT_TEMPLATE
            ),
            max_context_tokens=get_config_value(
                config, "retrieval.max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS
            ),
            config=config,
            config_path=config_path,
        )

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query)
        _, results = self.vector_store.search(query_embedding, k=top_k or self.top_k)
        return results

    def generate(
        self,
        query: str,
        context: Optional[list[RetrievalResult]] = None,
        llm_override: Optional[BaseLLM] = None,
    ) -> str:
        context = context or self.retrieve(query)
        llm_to_use = llm_override or self.llm
        model = getattr(llm_to_use, "model", "gpt-4")
        overhead = _count_tokens(
            self.context_template.format(context="", question=query), model
        )
        available = self.max_context_tokens - overhead

        context_text = ""
        current = 0
        truncated = False

        for doc in context:
            tokens = _count_tokens(doc.text, model)
            if current + tokens <= available:
                context_text = (
                    f"{context_text}\n\n{doc.text}" if context_text else doc.text
                )
                current += tokens
            else:
                truncated = True
                break

        if truncated:
            logger.warning(
                "Context truncated to %d tokens (limit: %d)",
                current,
                self.max_context_tokens,
            )

        return llm_to_use.generate(
            self.context_template.format(context=context_text, question=query)
        )

    def query(
        self, query: str, llm_override: Optional[BaseLLM] = None
    ) -> dict[str, Any]:
        context = self.retrieve(query)
        return {
            "response": self.generate(query, context, llm_override),
            "context": context,
        }


def get_retrieval_pipeline(
    config_path: Path = Path("config.toml"),
) -> RetrievalPipeline:
    config = load_config(config_path)
    return RetrievalPipeline.from_config(config, config_path)
