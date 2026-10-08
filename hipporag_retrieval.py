"""Native graph retrieval used by the HippoRAG baseline."""

from __future__ import annotations

import re
from contextlib import closing, contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import cast

import numpy as np
from hipporag import HippoRAG
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm.base import BaseLLM, LLMConfig
from hipporag.llm.openai_gpt import cache_response
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import QuerySolution
from openai import OpenAI
from openai.types.responses import ResponseTextConfig
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.shared import Reasoning, ResponseFormatJSONObject, ResponseFormatText
from openai.types.shared_params import Reasoning as ReasoningParams

from benchmark import EmbeddingSettings, Message, RetrievedContext, Settings, Usage
from cost import MeteredClient, indexing


class ResponsesLLM(BaseLLM):
    def __init__(self, config: BaseConfig, settings: Settings) -> None:
        super().__init__(config)
        self.llm_name = settings.llm.model
        self.settings = settings
        if settings.hipporag is None:
            raise ValueError("HippoRAG requires its max_output_tokens setting.")
        self.max_output_tokens = settings.hipporag.max_output_tokens
        self.usage: list[Usage] = []
        cache_dir = Path(config.save_dir) / "responses_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_name = str(cache_dir / "responses-v2.sqlite")
        self._init_llm_config()
        self.client = OpenAI(
            timeout=settings.llm.timeout,
            max_retries=0,
            http_client=MeteredClient(timeout=settings.llm.timeout),
        )

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()
        self.llm_config.generate_params = self.settings.model_dump()

    @cache_response
    def infer(self, messages, **kwargs):
        # HippoRAG's small text budgets omit reasoning tokens. Reserve room for both.
        json_output = kwargs.get("response_format") is not None
        if json_output:
            # Upstream's NER prompt asks for a list, but its parser requires an object.
            messages = [
                *messages,
                Message(
                    role="user",
                    content=(
                        "Return a JSON object matching the example's keys and structure. "
                        "For named entity extraction, use the key named_entities with a list of strings. "
                        "For triple extraction, use the key triples with a list of three-string lists."
                    ),
                ).model_dump(),
            ]
        response = self.client.responses.create(
            model=self.llm_name,
            input=messages,
            reasoning=cast(
                ReasoningParams,
                Reasoning(effort=self.settings.llm.reasoning_effort).model_dump(
                    exclude_none=True
                ),
            ),
            max_output_tokens=max(
                kwargs.get("max_new_tokens", 0),
                self.max_output_tokens,
            ),
            store=False,
            text=cast(
                ResponseTextConfigParam,
                ResponseTextConfig(
                    format=ResponseFormatJSONObject(type="json_object")
                    if json_output
                    else ResponseFormatText(type="text")
                ).model_dump(exclude_none=True),
            ),
        )
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(
                f"HippoRAG generation did not complete: {response.status}"
            )
        if response.usage is None:
            raise RuntimeError("HippoRAG generation returned no token usage.")
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )
        self.usage.append(usage)
        response_text = response.output_text
        if "fact_before_filter" in messages[-1]["content"]:
            # The upstream DSPy parser requires spaces in these field delimiters.
            response_text = re.sub(
                r"\[\[\s*##\s*(fact_after_filter|completed)\s*##\s*\]\]",
                r"[[ ## \1 ## ]]",
                response_text,
            )
        # The upstream cache adds its cache-hit flag for extraction and QA callers.
        return [response_text, usage.model_dump()]

    def close(self) -> None:
        self.client.close()


class ConfiguredEmbeddings(OpenAIEmbeddingModel):
    def __init__(self, config: BaseConfig, settings: EmbeddingSettings) -> None:
        self.dimensions = settings.dimensions
        super().__init__(config)
        self.client.close()
        self.client = OpenAI(
            timeout=settings.timeout,
            max_retries=0,
            http_client=MeteredClient(timeout=settings.timeout),
        )

    def encode(self, texts):
        response = self.client.embeddings.create(
            model=self.request_model_name,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        rows = sorted(response.data, key=lambda item: item.index)
        if [row.index for row in rows] != list(range(len(texts))):
            raise RuntimeError("Embedding response did not match the input documents.")
        vectors = np.asarray([row.embedding for row in rows], dtype=np.float32)
        if list(vectors.shape) != [len(texts), self.dimensions]:
            raise RuntimeError("Embedding response had unexpected dimensions.")
        self.last_usage = response.usage.model_dump()
        return vectors


@contextmanager
def setup(settings: Settings, index_dir: Path, dataset: str):
    if settings.hipporag is None:
        raise ValueError("HippoRAG requires its max_output_tokens setting.")
    config = BaseConfig(
        save_dir=str(index_dir),
        dataset="hotpotqa" if dataset == "hotpotqa" else "musique",
        llm_name=settings.llm.model,
        embedding_model_name=settings.embedding.model,
        embedding_provider="openai",
        embedding_request_timeout=settings.embedding.timeout,
        max_retry_attempts=0,
        temperature=None,
        max_new_tokens=settings.hipporag.max_output_tokens,
        retrieval_top_k=10,
        qa_top_k=10,
        response_format=ResponseFormatJSONObject(type="json_object").model_dump(),
    )
    with (
        closing(ResponsesLLM(config, settings)) as llm,
        closing(ConfiguredEmbeddings(config, settings.embedding)) as embeddings,
    ):
        index_dir.mkdir(parents=True, exist_ok=True)
        yield SimpleNamespace(config=config, llm=llm, embeddings=embeddings)


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
        index_marker = Path(config.save_dir) / "index.complete"
        should_index = corpus_key not in indexed_corpora and not index_marker.exists()
        if should_index:
            with indexing():
                rag.index(docs=list(documents))
            index_marker.write_text(corpus_key)
        indexed_corpora.add(corpus_key)
        construction_seconds = perf_counter() - started if should_index else 0.0
        started = perf_counter()
        solution = cast(
            list[QuerySolution], rag.retrieve(queries=[question], num_to_retrieve=top_k)
        )[0]
    return RetrievedContext(
        passages=solution.docs,
        scores=solution.doc_scores.tolist(),
        construction_seconds=construction_seconds,
        retrieval_seconds=perf_counter() - started,
    )


if __name__ == "__main__":
    from benchmark import main

    main(default_method="hipporag")
