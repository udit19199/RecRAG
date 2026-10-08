"""Compare RAG, RAPTOR, HippoRAG, and ontology agentic GraphRAG on shared questions."""

from __future__ import annotations

import argparse
import logging
import tomllib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal

import tiktoken
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from cost import MeteredClient, Pricing, QuestionCost, measure
from dataset import DATASET_SOURCES, SourcePage, get_source, load_hotpotqa_corpus
from evals.qa_metrics import (
    benchmark_prompt,
    hotpotqa_f1,
    score_novelqa_answer,
    score_qa_answer,
)

ROOT = Path(__file__).resolve().parent
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20
TOP_K = 10
Method = Literal["rag", "hipporag", "raptor", "graphrag"]


class Message(BaseModel):
    role: str
    content: str


class RetrievalOptions(BaseModel):
    top_k: PositiveInt = TOP_K
    rerank: bool = False
    rerank_candidate_k: PositiveInt = 20
    rerank_top_k: PositiveInt = TOP_K
    ircot: bool = False
    max_steps: PositiveInt = 2


class ReasoningStep(BaseModel):
    retrieval_query: str
    text: str


IRCOT_INSTRUCTION = (
    "You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning "
    "across multiple documents. This task is illustrated through demonstrations, each consisting of a document "
    "set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought "
    "for current step, DON'T generate the whole thoughts at once! If you reach what you believe to be the final step, "
    'start with "So the answer is:".'
)
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
NOVELQA_QA_INSTRUCTION = (
    "Answer only from the provided context. If the question lists options, "
    "return only the letter of the best option. Otherwise, give a short answer."
)


class RetrievedContext(BaseModel):
    passages: list[str]
    scores: list[float]
    construction_seconds: float
    retrieval_seconds: float


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


class CorpusManifest(BaseModel):
    start: int
    records: int
    page_entries: int
    chunks: int
    sha256: str


class Settings(BaseModel):
    llm: LLMSettings
    embedding: EmbeddingSettings
    hipporag: HippoSettings | None = None
    pricing: Pricing = Field(default_factory=Pricing)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    input_source: str = "official"
    paper_data: Path | None = None
    upstream_commit: str = "d2a0c0c0deb0903d60338d3c416ccd6f9544267c"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str = "stop"


class SavedRetrieval(BaseModel):
    method: Method
    dataset: str
    record_id: str
    question_type: str | None
    question: str
    canonical_answer: str | None
    expected_answers: list[str]
    supporting_sentences: list[str]
    retrieval: RetrievedContext
    index_dir: Path
    storage_mb: float
    llm_usage: list[Usage]
    question_cost: QuestionCost | None = None
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)
    complexity: str | None = None
    corpus_sha256: str | None = None


class BaselineResult(BaseModel):
    method: Method
    dataset: str
    record_id: str
    question_type: str | None
    question: str
    prompt: str
    answer: str
    embedding_model: str
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    chunk_unit: Literal["tokens", "characters"] = "tokens"
    top_k: int = TOP_K
    expected_answers: list[str]
    supporting_sentences: list[str]
    retrieved_context: list[str]
    retrieval_scores: list[float]
    exact_match: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    overlap_accuracy: float | None
    retrieval_accuracy: float | None
    retrieved_tokens: int
    storage_mb: float
    construction_seconds: float
    retrieval_seconds: float
    answer_seconds: float
    retrieval_and_answer_seconds: float
    llm_usage: list[Usage]
    question_cost: QuestionCost | None = None
    estimated_question_cost_usd: float | None = None
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)
    complexity: str | None = None
    corpus_sha256: str | None = None


class CategorySummary(BaseModel):
    question_type: str
    count: int
    accuracy: float | None
    complexity: str | None = None


class BaselineSummary(BaseModel):
    method: Method
    dataset: str
    count: int
    precision: float | None
    recall: float | None
    f1: float | None
    exact_match: float | None
    overlap_accuracy: float | None
    retrieval_accuracy: float | None
    total_construction_seconds: float
    mean_retrieval_seconds: float
    mean_answer_seconds: float
    mean_retrieved_tokens: float
    total_storage_mb: float
    categories: list[CategorySummary]
    total_question_cost_usd: float | None = None
    mean_question_cost_usd: float | None = None
    mean_retrieval_and_answer_seconds: float = 0
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)
    novelqa_categories: list[CategorySummary] = Field(default_factory=list)


def split_pages(pages: Sequence[SourcePage]) -> list[str]:
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import MetadataMode

    # Both retrievers must see the same titled chunks before their indexes diverge.
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[str] = []
    for page in pages:
        document = Document(text="\n\n".join(page.passages))
        document.metadata["title"] = page.title
        chunks.extend(
            f"{page.title}\n{node.get_content(metadata_mode=MetadataMode.NONE)}"
            for node in splitter.get_nodes_from_documents([document])
        )
    return chunks


def index_size_mb(index_dir: Path) -> float:
    return (
        sum(
            path.stat().st_size
            for path in index_dir.rglob("*")
            if path.is_file() and "responses_cache" not in path.parts
        )
        / 1_000_000
    )


def mean_score(values: Sequence[float | None]) -> float | None:
    scored = [value for value in values if value is not None]
    return fmean(scored) if scored else None


def refine_retrieval(
    question: str,
    initial: RetrievedContext,
    search: Callable[[str], RetrievedContext],
    llm,
    options: RetrievalOptions,
    reranker,
) -> list[ReasoningStep]:
    """Keep the upstream IRCoT union, or rerank once to ten evidence items."""
    if options.rerank:
        if initial.passages:
            scores = reranker.predict(
                [[question, passage] for passage in initial.passages]
            )
            order = sorted(
                range(len(scores)), key=lambda index: float(scores[index]), reverse=True
            )
            initial.passages = [
                initial.passages[index] for index in order[: options.rerank_top_k]
            ]
            initial.scores = [
                float(scores[index]) for index in order[: options.rerank_top_k]
            ]
        return []
    if not options.ircot:
        return []
    steps: list[ReasoningStep] = []
    seen = {" ".join(passage.split()) for passage in initial.passages}
    for _ in range(options.max_steps):
        context = "\n\n".join(initial.passages)
        previous = " ".join(step.text for step in steps)
        text = llm.infer(
            [
                Message(role="system", content=IRCOT_INSTRUCTION).model_dump(),
                Message(
                    role="user",
                    content=f"Context:\n{context}\n\nQuestion: {question}\nThought: {previous}",
                ).model_dump(),
            ]
        )[0]
        steps.append(
            ReasoningStep(
                retrieval_query=question if not steps else steps[-1].text, text=text
            )
        )
        if "So the answer is:" in text:
            break
        # Upstream retrieves after each nonfinal thought, including the last step.
        added = search(text)
        for passage, score in zip(added.passages, added.scores, strict=True):
            key = " ".join(passage.split())
            if key not in seen:
                seen.add(key)
                initial.passages.append(passage)
                initial.scores.append(score)
    return steps


def search_method(
    question: str,
    *,
    method: Method,
    pipeline,
    documents: list[str],
    pages: Sequence[SourcePage],
    corpus_key: str,
    indexed_corpora: set[str],
    index_dir: Path,
    top_k: int,
) -> RetrievedContext:
    if method == "hipporag":
        from hipporag_retrieval import retrieve

        return retrieve(
            config=pipeline.config,
            documents=documents,
            corpus_key=corpus_key,
            indexed_corpora=indexed_corpora,
            llm=pipeline.llm,
            embeddings=pipeline.embeddings,
            question=question,
            top_k=top_k,
        )
    if method == "raptor":
        from raptor_retrieval import retrieve as retrieve_raptor

        return retrieve_raptor(
            documents=documents,
            question=question,
            index_dir=index_dir,
            embeddings=pipeline.embeddings,
            llm=pipeline.llm,
            top_k=top_k,
        )
    if method == "graphrag":
        from graphrag_retrieval import retrieve as retrieve_graph

        return retrieve_graph(
            rag=pipeline.rag,
            pages=pages,
            question=question,
            index_dir=index_dir,
            top_k=top_k,
        )
    from rag_retrieval import retrieve as retrieve_dense

    return retrieve_dense(
        documents=documents,
        question=question,
        index_dir=index_dir,
        embeddings=pipeline.embeddings,
        top_k=top_k,
    )


def run(
    method: Method,
    dataset: str,
    start: int,
    records: int,
    options: RetrievalOptions | None = None,
    paper_data: Path | None = None,
) -> Path:
    settings = Settings.model_validate(
        tomllib.loads((ROOT / "config.toml").read_text())
    )
    if options is not None:
        settings.retrieval = options
    options = settings.retrieval
    settings.input_source = "paper_processed" if paper_data is not None else "official"
    settings.paper_data = paper_data
    if options.ircot and options.rerank:
        raise ValueError(
            "Use separate IRCoT and reranking runs, as in the paper's comparisons."
        )
    reranker = None
    if options.rerank:
        from sentence_transformers import CrossEncoder

        reranker = CrossEncoder("BAAI/bge-reranker-large")
    load_dotenv(ROOT / ".env", override=True)
    source = get_source(dataset, paper_data=paper_data)
    run_dir = ROOT / "runs" / method / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    (run_dir / "settings.json").write_text(settings.model_dump_json(indent=2))
    retrieval_path = run_dir / "retrieval.jsonl"
    selected_records = [
        source.load_record(index) for index in range(start, start + records)
    ]
    shared_documents: list[str] = []
    if dataset == "hotpotqa":
        corpus_pages = load_hotpotqa_corpus()
        shared_documents = split_pages(corpus_pages)
        corpus_key = sha256("\n\n".join(shared_documents).encode()).hexdigest()
        (run_dir / "corpus.json").write_text(
            CorpusManifest(
                start=start,
                records=len(selected_records),
                page_entries=len(corpus_pages),
                chunks=len(shared_documents),
                sha256=corpus_key,
            ).model_dump_json(indent=2)
        )
    cached_pages: Sequence[SourcePage] | None = None
    documents: list[str] = []
    indexed_corpora: set[str] = set()
    for record in selected_records:
        if dataset == "hotpotqa":
            documents = shared_documents
        elif record.pages is not cached_pages:
            documents = split_pages(record.pages)
            cached_pages = record.pages
        corpus_key = sha256("\n\n".join(documents).encode()).hexdigest()
        index_key = sha256(
            (
                corpus_key
                + settings.embedding.model_dump_json()
                + settings.llm.model_dump_json()
                + method
                + (
                    "paper-chunks256-overlap20-v3-json-object"
                    if method == "hipporag"
                    else "paper-chunks256-overlap20-v5-reference-clustering"
                    if method == "raptor"
                    else "paper-chunks256-overlap20-v2"
                )
                + (
                    settings.hipporag.model_dump_json()
                    if method == "hipporag" and settings.hipporag
                    else ""
                )
            ).encode()
        ).hexdigest()
        corpus_dir = ROOT / "runs" / method / "indexes" / index_key
        if method == "hipporag":
            from hipporag_retrieval import setup
        elif method == "graphrag":
            from graphrag_retrieval import setup
        else:
            from rag_retrieval import setup
        with setup(settings, corpus_dir, dataset) as pipeline:
            llm = pipeline.llm
            index_dir = corpus_dir
            search = partial(
                search_method,
                method=method,
                pipeline=pipeline,
                documents=documents,
                pages=corpus_pages if dataset == "hotpotqa" else record.pages,
                corpus_key=corpus_key,
                indexed_corpora=indexed_corpora,
                index_dir=index_dir,
                top_k=(options.rerank_candidate_k if options.rerank else options.top_k),
            )

            with measure(settings.pricing) as question_cost:
                started = perf_counter()
                retrieval = search(record.question)
                steps = refine_retrieval(
                    record.question, retrieval, search, llm, options, reranker
                )
                retrieval.retrieval_seconds = (
                    perf_counter() - started - retrieval.construction_seconds
                )
            saved = SavedRetrieval(
                method=method,
                dataset=dataset,
                record_id=record.id,
                question_type=getattr(record, "question_type", None),
                question=record.question,
                canonical_answer=record.answer,
                expected_answers=list(record.answer_aliases()),
                supporting_sentences=list(record.supporting_sentences()),
                retrieval=retrieval,
                index_dir=index_dir,
                storage_mb=index_size_mb(index_dir),
                llm_usage=[
                    Usage(
                        prompt_tokens=call.input_tokens,
                        completion_tokens=call.output_tokens,
                        total_tokens=call.input_tokens + call.output_tokens,
                    )
                    for call in question_cost.calls
                    if call.kind == "responses"
                ],
                question_cost=question_cost,
                reasoning_steps=steps,
                options=options,
                complexity=getattr(record, "complexity", None),
                corpus_sha256=corpus_key,
            )
        with retrieval_path.open("a", encoding="utf-8") as file:
            file.write(saved.model_dump_json() + "\n")
        print(f"{method}: saved retrieval for {record.id}", flush=True)
    return answer_saved(retrieval_path)


def answer_saved(retrieval_path: Path) -> Path:
    """Use the same answer client for both methods, reading only saved evidence."""
    from openai import OpenAI

    from rag_retrieval import ResponsesLLM

    run_dir = retrieval_path.parent
    settings = Settings.model_validate_json((run_dir / "settings.json").read_text())
    load_dotenv(ROOT / ".env", override=True)
    saved_records = [
        SavedRetrieval.model_validate_json(line)
        for line in retrieval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not saved_records:
        raise ValueError("The retrieval file contains no questions.")
    method = saved_records[0].method
    dataset = saved_records[0].dataset
    if any(
        saved.method != method
        or saved.dataset != dataset
        or saved.options != saved_records[0].options
        for saved in saved_records
    ):
        raise ValueError(
            "The retrieval file mixes methods, datasets, or retrieval options."
        )
    results_path = run_dir / "results.jsonl"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results: list[BaselineResult] = []
    index_dirs = {saved.index_dir for saved in saved_records}
    with OpenAI(
        timeout=settings.llm.timeout,
        max_retries=0,
        http_client=MeteredClient(timeout=settings.llm.timeout),
    ) as client:
        llm = ResponsesLLM(client, settings)
        for index, saved in enumerate(saved_records):
            retrieved_context = saved.retrieval.passages
            retrieval_scores = saved.retrieval.scores
            construction_seconds = saved.retrieval.construction_seconds
            retrieval_seconds = saved.retrieval.retrieval_seconds
            storage_mb = saved.storage_mb
            usage_start = len(llm.usage)
            context = "\n\n".join(retrieved_context)
            prompt = benchmark_prompt(context, saved.question)
            started = perf_counter()
            with measure(settings.pricing) as answer_cost:
                answer = llm.infer(
                    [
                        Message(
                            role="system",
                            content=NEWS_QA_INSTRUCTION
                            if dataset == "multihop_rag"
                            else NOVELQA_QA_INSTRUCTION
                            if dataset == "novelqa"
                            else QA_INSTRUCTION,
                        ).model_dump(),
                        Message(role="user", content=prompt).model_dump(),
                    ]
                )[0]
            answer_seconds = perf_counter() - started
            question_cost = (
                QuestionCost(
                    calls=saved.question_cost.calls + answer_cost.calls,
                    missing_usage_calls=saved.question_cost.missing_usage_calls
                    + answer_cost.missing_usage_calls,
                )
                if saved.question_cost is not None
                else None
            )
            expected_answers = list(saved.expected_answers)
            canonical_answer = saved.canonical_answer
            metrics = (
                score_novelqa_answer(answer, canonical_answer, prompt)
                if dataset == "novelqa" and canonical_answer is not None
                else score_qa_answer(
                    answer,
                    expected_answers,
                    canonical_answer,
                    prompt,
                    multihop_rag=dataset == "multihop_rag",
                )
                if canonical_answer is not None and expected_answers
                else None
            )
            if dataset == "hotpotqa" and metrics is not None:
                precision, recall, f1 = hotpotqa_f1(answer, expected_answers)
                metrics = metrics.model_copy(
                    update={"precision": precision, "recall": recall, "f1": f1}
                )
            result = BaselineResult(
                method=method,
                dataset=dataset,
                record_id=saved.record_id,
                question_type=saved.question_type,
                question=saved.question,
                prompt=prompt,
                answer=answer,
                embedding_model=settings.embedding.model,
                chunk_size=1000 if method == "graphrag" else CHUNK_SIZE,
                chunk_overlap=100 if method == "graphrag" else CHUNK_OVERLAP,
                chunk_unit="characters" if method == "graphrag" else "tokens",
                top_k=saved.options.top_k,
                expected_answers=expected_answers,
                supporting_sentences=list(saved.supporting_sentences),
                retrieved_context=retrieved_context,
                retrieval_scores=retrieval_scores,
                exact_match=metrics.exact_match if metrics is not None else None,
                precision=metrics.precision if metrics is not None else None,
                recall=metrics.recall if metrics is not None else None,
                f1=metrics.f1 if metrics is not None else None,
                # The paper uses any shared word for MultiHop-RAG accuracy.
                overlap_accuracy=(
                    metrics.overlap_accuracy if metrics is not None else None
                ),
                # Its retrieval check searches the entire unnormalized QA prompt.
                retrieval_accuracy=(
                    metrics.retrieval_accuracy if metrics is not None else None
                ),
                retrieved_tokens=len(tokenizer.encode(context)),
                storage_mb=storage_mb,
                construction_seconds=construction_seconds,
                retrieval_seconds=retrieval_seconds,
                answer_seconds=answer_seconds,
                retrieval_and_answer_seconds=retrieval_seconds + answer_seconds,
                llm_usage=saved.llm_usage + llm.usage[usage_start:],
                question_cost=question_cost,
                estimated_question_cost_usd=question_cost.estimated_usd
                if question_cost is not None
                else None,
                reasoning_steps=saved.reasoning_steps,
                options=saved.options,
                complexity=saved.complexity,
                corpus_sha256=saved.corpus_sha256,
            )
            results.append(result)
            with results_path.open(
                "w" if index == 0 else "a", encoding="utf-8"
            ) as file:
                file.write(result.model_dump_json() + "\n")
            print(
                f"{method} record {index}: "
                + (
                    f"EM={result.exact_match:.2f}, F1={result.f1:.2f}"
                    if result.exact_match is not None and result.f1 is not None
                    else "unscored (no gold answer)"
                ),
                flush=True,
            )
    summary = BaselineSummary(
        method=method,
        dataset=dataset,
        count=len(results),
        options=saved_records[0].options,
        total_question_cost_usd=(
            sum(result.estimated_question_cost_usd or 0 for result in results)
            if all(result.estimated_question_cost_usd is not None for result in results)
            else None
        ),
        mean_question_cost_usd=(
            fmean(result.estimated_question_cost_usd or 0 for result in results)
            if all(result.estimated_question_cost_usd is not None for result in results)
            else None
        ),
        mean_retrieval_and_answer_seconds=fmean(
            result.retrieval_and_answer_seconds for result in results
        ),
        precision=mean_score([result.precision for result in results]),
        recall=mean_score([result.recall for result in results]),
        f1=mean_score([result.f1 for result in results]),
        exact_match=mean_score([result.exact_match for result in results]),
        overlap_accuracy=mean_score([result.overlap_accuracy for result in results]),
        retrieval_accuracy=mean_score(
            [result.retrieval_accuracy for result in results]
        ),
        total_construction_seconds=sum(
            result.construction_seconds for result in results
        ),
        mean_retrieval_seconds=fmean(result.retrieval_seconds for result in results),
        mean_answer_seconds=fmean(result.answer_seconds for result in results),
        mean_retrieved_tokens=fmean(result.retrieved_tokens for result in results),
        total_storage_mb=sum(
            next(
                saved.storage_mb
                for saved in reversed(saved_records)
                if saved.index_dir == path
            )
            for path in index_dirs
        ),
        categories=[
            CategorySummary(
                question_type=question_type,
                count=sum(result.question_type == question_type for result in results),
                accuracy=mean_score(
                    [
                        result.overlap_accuracy
                        for result in results
                        if result.question_type == question_type
                    ]
                ),
            )
            for question_type in sorted(
                {result.question_type for result in results if result.question_type}
            )
        ],
        novelqa_categories=[
            CategorySummary(
                question_type=aspect,
                complexity=complexity,
                count=sum(
                    result.question_type == aspect and result.complexity == complexity
                    for result in results
                ),
                accuracy=mean_score(
                    [
                        result.exact_match
                        for result in results
                        if result.question_type == aspect
                        and result.complexity == complexity
                    ]
                ),
            )
            for complexity in sorted(
                {result.complexity for result in results if result.complexity}
            )
            for aspect in sorted(
                {result.question_type for result in results if result.question_type}
            )
            if dataset == "novelqa"
            and any(
                result.question_type == aspect and result.complexity == complexity
                for result in results
            )
        ],
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))
    return results_path


def main(default_method: Method = "hipporag") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["rag", "hipporag", "raptor", "graphrag"],
        default=default_method,
    )
    parser.add_argument(
        "--dataset",
        choices=[source.name for source in DATASET_SOURCES],
        default="hotpotqa",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--topk", type=int, default=TOP_K)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--rerank-topk", type=int, default=TOP_K)
    parser.add_argument("--max-steps", type=int, default=2)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--rerank", action="store_true")
    variants.add_argument("--ircot", action="store_true")
    parser.add_argument(
        "--paper-data",
        type=Path,
        help="Directory containing the authors' processed dataset JSON files.",
    )
    parser.add_argument(
        "--answer-from",
        type=Path,
        metavar="RETRIEVAL_JSONL",
        help="Generate and score answers from saved retrieval, without searching again.",
    )
    args = parser.parse_args()
    if args.start < 0 or args.records < 1:
        parser.error("--start must be nonnegative and --records must be positive")
    if min(args.topk, args.rerank_candidates, args.rerank_topk, args.max_steps) < 1:
        parser.error(
            "--topk, --rerank-candidates, --rerank-topk, and --max-steps must be positive"
        )
    logging.basicConfig(level=logging.WARNING)
    print(
        answer_saved(args.answer_from)
        if args.answer_from is not None
        else run(
            args.method,
            args.dataset,
            args.start,
            args.records,
            RetrievalOptions(
                top_k=args.topk,
                rerank=args.rerank,
                rerank_candidate_k=args.rerank_candidates,
                rerank_top_k=args.rerank_topk,
                ircot=args.ircot,
                max_steps=args.max_steps,
            ),
            args.paper_data,
        )
    )


if __name__ == "__main__":
    # Adapters import benchmark's model classes. Use that same module for the CLI.
    from benchmark import main as cli

    cli()
