"""BenchmarkRunner — query candidates and score (BM-5 parallel, BM-6 skip failures)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from benchmark.scoring import build_summary, composite_score
from benchmark.suite import build_benchmark_suite
from orchestration.models import (
    BenchmarkSummary,
    PipelineCandidate,
    Requirements,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)


async def _query_retrieval(
    client: httpx.AsyncClient,
    query: str,
    spec: PipelineCandidate,
    api_key: str | None,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if api_key:
        headers["RecRAG-API-Key"] = api_key
    body = {
        "query": query,
        "embedding": {
            "provider": spec.spec.ingestion.embedding.provider,
            "model": spec.spec.ingestion.embedding.model,
        },
        "llm": {
            "provider": spec.spec.retrieval.llm.provider,
            "model": spec.spec.retrieval.llm.model,
        },
    }
    if spec.spec.ingestion.vision:
        body["vision"] = {
            "provider": spec.spec.ingestion.vision.provider,
            "model": spec.spec.ingestion.vision.model,
        }
    if spec.spec.collection_name:
        body["collection_name"] = spec.spec.collection_name
    if spec.spec.retrieval.top_k:
        body["top_k"] = spec.spec.retrieval.top_k
    body["pipeline_type"] = spec.spec.architecture.value
    resp = await client.post("/query", json=body, headers=headers, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


async def _eval_query(
    query: str,
    contexts: list[str],
    response: str,
) -> dict[str, float]:
    import asyncio

    from evaluation.ragas_eval import get_evaluator

    evaluator = get_evaluator()
    return await asyncio.to_thread(
        evaluator.evaluate_query, query, contexts, response
    )


async def benchmark_candidate(
    retrieval_url: str,
    candidate: PipelineCandidate,
    queries: list[str],
    weights: dict[str, float],
    api_key: str | None,
) -> tuple[PipelineCandidate, BenchmarkSummary] | None:
    totals: dict[str, float] = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    n = 0
    try:
        async with httpx.AsyncClient(base_url=retrieval_url) as client:
            for query in queries:
                result = await _query_retrieval(client, query, candidate, api_key)
                contexts = [c["text"] for c in result.get("context", [])]
                scores = await _eval_query(
                    query, contexts, result.get("response", "")
                )
                for k in totals:
                    totals[k] += scores.get(k, 0.0)
                n += 1
    except Exception as exc:
        logger.warning(
            "Benchmark failed for candidate %s: %s", candidate.spec.id, exc
        )
        return None

    if n == 0:
        return None
    avg = {k: v / n for k, v in totals.items()}
    summary = build_summary(avg, weights, candidates_evaluated=1)
    return candidate, summary


async def run_benchmark(
    requirements: Requirements,
    candidates: list[PipelineCandidate],
    weights: dict[str, float],
    retrieval_url: str,
    api_key: str | None = None,
) -> list[ScoredCandidate]:
    """BM-5: parallel benchmark; BM-6: skip failed candidates."""
    suite = build_benchmark_suite(
        requirements, [c.spec.architecture for c in candidates]
    )
    queries = suite.all_queries()
    if not queries:
        queries = ["What information do these documents contain?"]

    tasks = [
        benchmark_candidate(retrieval_url, c, queries, weights, api_key)
        for c in candidates
    ]
    results = await asyncio.gather(*tasks)

    scored: list[ScoredCandidate] = []
    for item in results:
        if item is None:
            continue
        candidate, summary = item
        scored.append(
            ScoredCandidate(
                spec=candidate.spec,
                benchmark_summary=summary,
                rank=0,
            )
        )

    scored.sort(
        key=lambda s: s.benchmark_summary.composite_score or 0.0,
        reverse=True,
    )
    for i, s in enumerate(scored, start=1):
        s.rank = i

    if len(scored) >= 2:
        top = scored[0].benchmark_summary.composite_score or 0.0
        second = scored[1].benchmark_summary.composite_score or 0.0
        scored[0].benchmark_summary.margin_over_runner_up = round(top - second, 4)

    return scored
