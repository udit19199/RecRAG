"""OrchestrationEngine — shortlist pipeline candidates (D2, D3)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import toml

from orchestration.collection_naming import get_collection_name_for_spec
from orchestration.constraints import eligible_architectures, metric_weight_key
from orchestration.cost import estimate_monthly_usd
from orchestration.customizer import build_candidate_specs, refine_with_llm
from orchestration.models import (
    ConfidenceLevel,
    ModelTier,
    PipelineCandidate,
    PipelineSpec,
    PreliminaryRecommendation,
    RagArchitecture,
    Requirements,
)
from orchestration.resolver import resolve_adapter

logger = logging.getLogger(__name__)


def _load_rules() -> dict[str, Any]:
    path = Path("data/recommendation_rules.toml")
    return toml.load(path) if path.exists() else {}


def _feasibility_score(spec: PipelineSpec, requirements: Requirements) -> float:
    score = 0.5
    if requirements.citations_required and spec.architecture == RagArchitecture.CITATION:
        score += 0.25
    if requirements.document_modality.value in ("image_heavy", "mixed"):
        if spec.architecture == RagArchitecture.MULTIMODAL:
            score += 0.25
    if requirements.budget_monthly_usd and spec.estimated_monthly_usd:
        if spec.estimated_monthly_usd <= requirements.budget_monthly_usd:
            score += 0.15
    return min(score, 1.0)


class OrchestrationEngine:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def shortlist(
        self,
        requirements: Requirements,
        provider_models: dict[str, list[str]],
        *,
        refine_llm: bool = True,
        run_id: str | None = None,
    ) -> list[PipelineCandidate]:
        rules = _load_rules()
        min_c = rules.get("shortlist", {}).get("min_candidates", 3)
        max_c = rules.get("shortlist", {}).get("max_candidates", 5)

        embed_provider = self.config.get("embedding", {}).get("provider", "gemini")
        llm_provider = self.config.get("llm", {}).get("provider", "gemini")

        embedding = resolve_adapter(
            provider_models, embed_provider, "embedding", ModelTier.FLAGSHIP
        )
        llm_mid = resolve_adapter(provider_models, llm_provider, "llm", ModelTier.MID)
        llm_flagship = resolve_adapter(
            provider_models, llm_provider, "llm", ModelTier.FLAGSHIP
        )
        vision = resolve_adapter(
            provider_models, f"vision_{embed_provider}", "vision", ModelTier.MID
        ) or resolve_adapter(provider_models, embed_provider, "llm", ModelTier.MID)

        if not embedding or not llm_mid:
            raise ValueError("Could not resolve embedding or LLM from providers")

        architectures = eligible_architectures(requirements)
        specs = build_candidate_specs(
            requirements, architectures, embedding, llm_mid, llm_flagship, vision
        )

        while len(specs) < min_c:
            # D4 backfill: retrieval variant
            variant = build_candidate_specs(
                requirements,
                [RagArchitecture.NAIVE],
                embedding,
                llm_flagship or llm_mid,
                llm_flagship,
                vision,
            )
            for v in variant:
                v.id = f"{v.id}-backfill-{len(specs)}"
                specs.append(v)
                if len(specs) >= min_c:
                    break

        specs = specs[:max_c]
        if refine_llm:
            specs = refine_with_llm(requirements, specs)

        candidates: list[PipelineCandidate] = []
        for spec in specs:
            spec.estimated_monthly_usd = estimate_monthly_usd(spec, requirements)
            spec.collection_name = get_collection_name_for_spec(
                self.config, spec, run_id
            )
            candidates.append(
                PipelineCandidate(
                    spec=spec,
                    feasibility_score=_feasibility_score(spec, requirements),
                )
            )
        candidates.sort(key=lambda c: c.feasibility_score, reverse=True)
        return candidates

    def preliminary(
        self,
        requirements: Requirements,
        candidates: list[PipelineCandidate],
    ) -> PreliminaryRecommendation:
        top = candidates[0] if candidates else None
        if not top:
            return PreliminaryRecommendation(
                architecture=RagArchitecture.NAIVE,
                rationale="Insufficient provider configuration to shortlist.",
            )
        return PreliminaryRecommendation(
            confidence=ConfidenceLevel.PRELIMINARY,
            architecture=top.spec.architecture,
            rationale=top.spec.rationale or "Preliminary shortlist based on requirements.",
            pros=top.spec.pros,
            cons=top.spec.cons,
            estimated_monthly_usd=top.spec.estimated_monthly_usd,
            candidates_preview=candidates,
        )

    @staticmethod
    def metric_weights(requirements: Requirements) -> dict[str, float]:
        rules = _load_rules()
        key = metric_weight_key(requirements)
        weights = rules.get("metric_weights", {}).get(key, {})
        if not weights:
            weights = rules.get("metric_weights", {}).get("default", {})
        return {k: float(v) for k, v in weights.items()}
