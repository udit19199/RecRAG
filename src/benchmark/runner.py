"""BenchmarkRunner — query candidates and score (BM-5 parallel, BM-6 skip failures)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from benchmark.scoring import build_summary
from benchmark.suite import build_benchmark_suite
from orchestration.models import (
    BenchmarkSuite,
    BenchmarkSummary,
    PipelineCandidate,
    Requirements,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)


def _queries_for_candidate(
    suite: BenchmarkSuite,
    architecture: str,
    user_queries: list[str],
) -> list[str]:
    """Domain/user queries plus architecture-specific probes for one candidate."""
    seen: set[str] = set()
    ordered: list[str] = []
    for q in suite.domain_queries + list(user_queries):
        if q.strip() and q not in seen:
            seen.add(q)
            ordered.append(q)
    for q in suite.architecture_probes.get(architecture, []):
        if q.strip() and q not in seen:
            seen.add(q)
            ordered.append(q)
    if not ordered:
        return ["What information do these documents contain?"]
    return ordered


async def _query_retrieval(
    client: httpx.AsyncClient,
    query: str,
    spec: PipelineCandidate,
) -> dict[str, Any]:
    body: dict[str, Any] = {
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
    resp = await client.post("/query", json=body, timeout=120.0)
    resp.raise_for_status()
    return resp.json()


async def _eval_query(
    query: str,
    contexts: list[str],
    response: str,
) -> dict[str, float]:
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
) -> tuple[tuple[PipelineCandidate, BenchmarkSummary] | None, str | None]:
    totals: dict[str, float] = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    n = 0
    last_error: str | None = None
    try:
        async with httpx.AsyncClient(base_url=retrieval_url) as client:
            for query in queries:
                try:
                    result = await _query_retrieval(client, query, candidate)
                    contexts = [c["text"] for c in result.get("context", [])]
                    scores = await _eval_query(
                        query, contexts, result.get("response", "")
                    )
                    for k in totals:
                        totals[k] += scores.get(k, 0.0)
                    n += 1
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Benchmark query failed for candidate %s: %s",
                        candidate.spec.id,
                        exc,
                    )
    except Exception as exc:
        last_error = str(exc)
        logger.warning(
            "Benchmark failed for candidate %s: %s", candidate.spec.id, exc
        )
        return None, last_error

    if n == 0:
        return None, last_error or "all benchmark queries failed"
    avg = {k: v / n for k, v in totals.items()}
    summary = build_summary(avg, weights, candidates_evaluated=1)
    return (candidate, summary), None


async def run_benchmark(
    requirements: Requirements,
    candidates: list[PipelineCandidate],
    weights: dict[str, float],
    retrieval_url: str,
) -> tuple[list[ScoredCandidate], list[str]]:
    """BM-5: parallel benchmark; BM-6: skip failed candidates."""
    suite = build_benchmark_suite(
        requirements, [c.spec.architecture for c in candidates]
    )

    tasks = [
        benchmark_candidate(
            retrieval_url,
            c,
            _queries_for_candidate(
                suite,
                c.spec.architecture.value,
                requirements.user_queries,
            ),
            weights,
        )
        for c in candidates
    ]
    results = await asyncio.gather(*tasks)

    scored: list[ScoredCandidate] = []
    failures: list[str] = []
    for candidate, (item, failure) in zip(candidates, results, strict=True):
        if item is None:
            failures.append(f"{candidate.spec.id}: {failure or 'unknown error'}")
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

    return scored, failures
