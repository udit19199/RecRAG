"""Post-resolution cost estimation (runs after PipelineSpec finalized — D12)."""

from __future__ import annotations

from orchestration.models import PipelineSpec, Requirements
from orchestration.pricing import model_unit_price


def estimate_monthly_usd(
    spec: PipelineSpec,
    requirements: Requirements,
) -> float:
    usage = requirements.usage
    pages = usage.document_pages or 100
    queries_day = usage.queries_per_day or 50
    q_tokens = usage.avg_query_tokens or 128
    a_tokens = usage.avg_answer_tokens or 512

    embed = spec.ingestion.embedding
    llm = spec.retrieval.llm
    embed_in = model_unit_price(embed.provider, embed.model, "input") or 0.0
    llm_in = model_unit_price(llm.provider, llm.model, "input") or 0.0
    llm_out = model_unit_price(llm.provider, llm.model, "output") or 0.0

    # Rough token estimates: 500 tokens per page ingest, one-time amortized over 30 days
    ingest_tokens = pages * 500
    monthly_tokens_in = (ingest_tokens / 30) + queries_day * (q_tokens + q_tokens) * 30
    monthly_tokens_out = queries_day * a_tokens * 30

    cost = (monthly_tokens_in / 1_000_000) * max(embed_in, llm_in)
    cost += (monthly_tokens_out / 1_000_000) * llm_out
    return round(cost, 2)
