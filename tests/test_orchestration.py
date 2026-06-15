"""Tests for orchestration engine and models."""

from __future__ import annotations

import uuid

from orchestration.constraints import eligible_architectures
from orchestration.cost import estimate_monthly_usd
from orchestration.customizer import build_candidate_specs
from orchestration.models import (
    AdapterRef,
    BenchmarkSummary,
    ConfidenceLevel,
    DocumentModality,
    PipelineBlueprint,
    RagArchitecture,
    Requirements,
)
from orchestration.engine import OrchestrationEngine
from benchmark.suite import build_benchmark_suite
from benchmark.scoring import composite_score


def test_eligible_architectures_citations() -> None:
    req = Requirements(
        use_case="legal compliance",
        citations_required=True,
        document_modality=DocumentModality.TEXT_HEAVY,
    )
    archs = eligible_architectures(req)
    assert RagArchitecture.CITATION in archs
    assert RagArchitecture.NAIVE not in archs


def test_shortlist_produces_candidates() -> None:
    engine = OrchestrationEngine({})
    req = Requirements(use_case="internal kb", citations_required=False)
    providers = {
        "gemini": ["gemini-embedding-001", "gemini-2.0-flash"],
        "openai": ["text-embedding-3-large", "gpt-4o", "gpt-4o-mini"],
    }
    candidates = engine.shortlist(req, providers, refine_llm=False, run_id="test-run")
    assert len(candidates) >= 3
    assert all(c.spec.collection_name for c in candidates)


def test_pipeline_blueprint_schema() -> None:
    spec = build_candidate_specs(
        Requirements(use_case="test"),
        [RagArchitecture.NAIVE],
        AdapterRef(provider="gemini", model="gemini-embedding-001"),
        AdapterRef(provider="gemini", model="gemini-2.0-flash"),
        None,
        None,
    )[0]
    spec.estimated_monthly_usd = estimate_monthly_usd(spec, Requirements(use_case="test"))
    blueprint = PipelineBlueprint(
        run_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        architecture=spec.architecture,
        ingestion=spec.ingestion,
        retrieval=spec.retrieval,
        rationale="test",
        pros=spec.pros,
        cons=spec.cons,
        benchmark_summary=BenchmarkSummary(
            confidence=ConfidenceLevel.MEASURED,
            candidates_evaluated=3,
            composite_score=0.8,
        ),
    )
    data = blueprint.model_dump(mode="json")
    assert data["architecture"] == "naive"
    assert "benchmark_summary" in data


def test_benchmark_suite_no_industry_templates() -> None:
    req = Requirements(use_case="policy documents", user_queries=["What is policy X?"])
    suite = build_benchmark_suite(req, [RagArchitecture.CITATION, RagArchitecture.NAIVE])
    queries = suite.all_queries()
    assert len(queries) >= 3
    assert "naive" in suite.architecture_probes


def test_composite_score_weights() -> None:
    weights = {"faithfulness": 0.5, "answer_relevancy": 0.5}
    scores = {"faithfulness": 0.8, "answer_relevancy": 0.6}
    assert composite_score(scores, weights) == 0.7
