"""Compare dense RAG and HippoRAG retrieval with shared inputs and answering."""

from __future__ import annotations

import argparse
import asyncio
import logging
import tomllib
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal, cast

import numpy as np
import tiktoken
from dotenv import load_dotenv
from hipporag.embedding_model.OpenAI import OpenAIEmbeddingModel
from hipporag.llm.base import BaseLLM, LLMConfig
from hipporag.llm.openai_gpt import cache_response
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.eval_utils import normalize_answer
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from openai import OpenAI
from openai.types.shared import Reasoning
from openai.types.shared_params import Reasoning as ReasoningParams
from pydantic import BaseModel, PositiveFloat, PositiveInt, PrivateAttr

from graphrag.datasets import DATASET_SOURCES, get_source
from hipporag import HippoRAG

ROOT = Path(__file__).resolve().parent
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 10
QA_INSTRUCTION = (
    "You are an AI assistant that answers questions using the provided context. "
    "Always base your answers only on the given context and provide clear, concise, "
    "and accurate responses. You must give ONLY the answer as a single word or "
    "phrase. Do not give additional information. If the provided information is "
    "insufficient to answer the question, respond 'Insufficient Information'."
)
NEWS_QA_INSTRUCTION = (
    "You are an AI assistant that answers questions using the provided context. "
    "Always base your answers strictly on the given context. Answer inference "
    "questions with an entity or short phrase. If the context is insufficient, "
    "respond ONLY with 'Insufficient Information'. For binary questions, "
    "respond ONLY with 'Yes' or 'No'."
)


class LLMSettings(BaseModel):
    model: Literal["gpt-6-luna"]
    reasoning_effort: Literal["medium"]
    timeout: PositiveFloat


class EmbeddingSettings(BaseModel):
    model: str
    dimensions: PositiveInt
    timeout: PositiveFloat


class HippoSettings(BaseModel):
    max_output_tokens: PositiveInt


class Settings(BaseModel):
    llm: LLMSettings
    embedding: EmbeddingSettings
    hipporag: HippoSettings


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str = "stop"


class ResponsesLLM(BaseLLM):
    def __init__(self, config: BaseConfig, settings: Settings) -> None:
        super().__init__(config)
        self.llm_name = settings.llm.model
        self.settings = settings
        self.usage: list[Usage] = []
        cache_dir = Path(config.save_dir) / "responses_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_name = str(cache_dir / "responses.sqlite")
        self._init_llm_config()
        self.client = OpenAI(timeout=settings.llm.timeout, max_retries=0)

    def _init_llm_config(self) -> None:
        self.llm_config = LLMConfig()
        self.llm_config.generate_params = self.settings.model_dump()

    @cache_response
    def infer(self, messages, **kwargs):
        # HippoRAG's small text budgets omit reasoning tokens. Reserve room for both.
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
                self.settings.hipporag.max_output_tokens,
            ),
            store=False,
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
        # The upstream cache adds its cache-hit flag for extraction and QA callers.
        return [response.output_text, usage.model_dump()]

    def close(self) -> None:
        self.client.close()


class ConfiguredEmbeddings(OpenAIEmbeddingModel):
    def __init__(self, config: BaseConfig, settings: EmbeddingSettings) -> None:
        self.dimensions = settings.dimensions
        super().__init__(config)

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


class BaselineResult(BaseModel):
    method: Literal["rag", "hipporag"]
    dataset: str
    record_id: str
    question_type: str | None
    question: str
    prompt: str
    answer: str
    embedding_model: str
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    top_k: int = TOP_K
    expected_answers: list[str]
    supporting_sentences: list[str]
    retrieved_context: list[str]
    retrieval_scores: list[float]
    exact_match: float
    precision: float
    recall: float
    f1: float
    overlap_accuracy: float
    retrieval_accuracy: float
    retrieved_tokens: int
    storage_mb: float
    construction_seconds: float
    retrieval_seconds: float
    answer_seconds: float
    retrieval_and_answer_seconds: float
    llm_usage: list[Usage]


class CategorySummary(BaseModel):
    question_type: str
    count: int
    accuracy: float


class BaselineSummary(BaseModel):
    method: Literal["rag", "hipporag"]
    dataset: str
    count: int
    precision: float
    recall: float
    f1: float
    exact_match: float
    overlap_accuracy: float
    retrieval_accuracy: float
    total_construction_seconds: float
    mean_retrieval_seconds: float
    mean_answer_seconds: float
    mean_retrieved_tokens: float
    total_storage_mb: float
    categories: list[CategorySummary]


def split_pages(pages) -> list[str]:
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    # Both retrievers must see the same titled chunks before their indexes diverge.
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [
        f"{node.metadata['title']}\n{node.text}"
        for page in pages
        for node in splitter.get_nodes_from_documents(
            [Document(text="\n\n".join(page.passages), metadata={"title": page.title})]
        )
    ]


def answer_scores(answer: str, aliases) -> list[float]:
    prediction = normalize_answer(answer).split()
    scores = []
    for alias in aliases:
        gold = normalize_answer(alias).split()
        overlap = sum((Counter(prediction) & Counter(gold)).values())
        precision = overlap / len(prediction) if prediction else 0.0
        recall = overlap / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
        scores.append([float(prediction == gold), precision, recall, f1])
    return max(scores, key=lambda score: score[3])


def index_size_mb(index_dir: Path) -> float:
    return (
        sum(
            path.stat().st_size
            for path in index_dir.rglob("*")
            if path.is_file() and "responses_cache" not in path.parts
        )
        / 1_000_000
    )


def run(
    method: Literal["rag", "hipporag"],
    dataset: str,
    start: int,
    records: int,
) -> Path:
    settings = Settings.model_validate(
        tomllib.loads((ROOT / "config.toml").read_text())
    )
    load_dotenv(ROOT / ".env", override=True)
    source = get_source(dataset)
    run_dir = ROOT / "runs" / method / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    (run_dir / "settings.json").write_text(settings.model_dump_json(indent=2))
    results_path = run_dir / "results.jsonl"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results: list[BaselineResult] = []
    cached_pages = None
    documents: list[str] = []
    indexed_corpora: set[str] = set()
    index_dirs: set[Path] = set()
    for index in range(start, start + records):
        record = source.load_record(index)
        if record.pages is not cached_pages:
            documents = split_pages(record.pages)
            cached_pages = record.pages
        corpus_key = sha256("\n\n".join(documents).encode()).hexdigest()
        config = BaseConfig(
            save_dir=str(run_dir / corpus_key),
            dataset="hotpotqa" if dataset == "hotpotqa" else "musique",
            llm_name=settings.llm.model,
            embedding_model_name=settings.embedding.model,
            embedding_provider="openai",
            embedding_request_timeout=settings.embedding.timeout,
            max_retry_attempts=0,
            temperature=None,
            max_new_tokens=settings.hipporag.max_output_tokens,
            retrieval_top_k=TOP_K,
            qa_top_k=TOP_K,
        )
        with (
            closing(ResponsesLLM(config, settings)) as llm,
            closing(ConfiguredEmbeddings(config, settings.embedding)) as embeddings,
        ):
            index_dir = Path(config.save_dir)
            index_dir.mkdir(parents=True, exist_ok=True)
            index_dirs.add(index_dir)
            started = perf_counter()
            if method == "hipporag":
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
                    solution = rag.retrieve(
                        queries=[record.question], num_to_retrieve=TOP_K
                    )[0]
                    retrieval_seconds = perf_counter() - started
                retrieved_context = solution.docs
                retrieval_scores = solution.doc_scores.tolist()
            else:
                embedding = LlamaEmbedding(embeddings, settings.embedding.model)
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
                hits = vector_index.as_retriever(similarity_top_k=TOP_K).retrieve(
                    record.question
                )
                retrieved_context = [hit.node.text for hit in hits]
                retrieval_scores = [float(hit.score or 0.0) for hit in hits]
                retrieval_seconds = perf_counter() - started
            storage_mb = index_size_mb(index_dir)
            context = "\n\n".join(retrieved_context)
            prompt = (
                f"Context:\n{context}\n\nQuestion:\n{record.question}? "
                "Please just give a short answer without explanation.\n\nAnswer:"
            )
            started = perf_counter()
            answer = llm.infer(
                [
                    {
                        "role": "system",
                        "content": NEWS_QA_INSTRUCTION
                        if dataset == "multihop_rag"
                        else QA_INSTRUCTION,
                    },
                    {"role": "user", "content": prompt},
                ]
            )[0]
            answer_seconds = perf_counter() - started
            expected_answers = list(record.answer_aliases())
            exact_match, precision, recall, f1 = answer_scores(answer, expected_answers)
            paper_tokens = set(answer.lower().replace(".", "").split())
            gold_tokens = set(record.answer.lower().replace(".", "").split())
            result = BaselineResult(
                method=method,
                dataset=dataset,
                record_id=record.id,
                question_type=getattr(record, "question_type", None),
                question=record.question,
                prompt=prompt,
                answer=answer,
                embedding_model=settings.embedding.model,
                expected_answers=expected_answers,
                supporting_sentences=list(record.supporting_sentences()),
                retrieved_context=retrieved_context,
                retrieval_scores=retrieval_scores,
                exact_match=exact_match,
                precision=precision,
                recall=recall,
                f1=f1,
                # The paper uses any shared word for MultiHop-RAG accuracy.
                overlap_accuracy=float(bool(paper_tokens & gold_tokens)),
                # Its retrieval check searches the entire unnormalized QA prompt.
                retrieval_accuracy=float(record.answer in prompt),
                retrieved_tokens=len(tokenizer.encode(context)),
                storage_mb=storage_mb,
                construction_seconds=construction_seconds,
                retrieval_seconds=retrieval_seconds,
                answer_seconds=answer_seconds,
                retrieval_and_answer_seconds=retrieval_seconds + answer_seconds,
                llm_usage=llm.usage,
            )
            results.append(result)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(result.model_dump_json() + "\n")
        print(
            f"{method} record {index}: EM={result.exact_match:.2f}, F1={result.f1:.2f}",
            flush=True,
        )
    summary = BaselineSummary(
        method=method,
        dataset=dataset,
        count=len(results),
        precision=fmean(result.precision for result in results),
        recall=fmean(result.recall for result in results),
        f1=fmean(result.f1 for result in results),
        exact_match=fmean(result.exact_match for result in results),
        overlap_accuracy=fmean(result.overlap_accuracy for result in results),
        retrieval_accuracy=fmean(result.retrieval_accuracy for result in results),
        total_construction_seconds=sum(
            result.construction_seconds for result in results
        ),
        mean_retrieval_seconds=fmean(result.retrieval_seconds for result in results),
        mean_answer_seconds=fmean(result.answer_seconds for result in results),
        mean_retrieved_tokens=fmean(result.retrieved_tokens for result in results),
        total_storage_mb=sum(index_size_mb(path) for path in index_dirs),
        categories=[
            CategorySummary(
                question_type=question_type,
                count=sum(result.question_type == question_type for result in results),
                accuracy=fmean(
                    result.overlap_accuracy
                    for result in results
                    if result.question_type == question_type
                ),
            )
            for question_type in sorted(
                {result.question_type for result in results if result.question_type}
            )
        ],
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rag", "hipporag"], default="hipporag")
    parser.add_argument(
        "--dataset",
        choices=[source.name for source in DATASET_SOURCES],
        default="hotpotqa",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--records", type=int, default=1)
    args = parser.parse_args()
    if args.start < 0 or args.records < 1:
        parser.error("--start must be nonnegative and --records must be positive")
    logging.basicConfig(level=logging.WARNING)
    print(run(args.method, args.dataset, args.start, args.records))


if __name__ == "__main__":
    main()
