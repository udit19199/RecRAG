"""Customizer LLM — tunes PipelineSpec slots and prose (D17)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import toml

from adapters import create_llm
from orchestration.models import (
    AdapterRef,
    IngestionSpec,
    PipelineSpec,
    RagArchitecture,
    Requirements,
    RetrievalSpec,
)

logger = logging.getLogger(__name__)


def _load_rules() -> dict[str, Any]:
    path = Path("data/recommendation_rules.toml")
    if path.exists():
        return toml.load(path)
    return {}


def _default_specs_for_architecture(
    architecture: RagArchitecture,
    embedding: AdapterRef,
    llm: AdapterRef,
    vision: AdapterRef | None,
) -> PipelineSpec:
    if architecture == RagArchitecture.CITATION:
        ingestion = IngestionSpec(
            extraction_mode="text_only",
            chunk_size=512,
            chunk_overlap=80,
            metadata_level="full",
            embedding=embedding,
        )
        retrieval = RetrievalSpec(llm=llm, top_k=6, reranker_enabled=False)
        pros = ["Strong citation metadata", "Fine-grained chunks"]
        cons = ["Higher ingest cost than naive"]
    elif architecture == RagArchitecture.MULTIMODAL:
        ingestion = IngestionSpec(
            extraction_mode="vision_assisted",
            chunk_size=768,
            chunk_overlap=64,
            metadata_level="full",
            embedding=embedding,
            vision=vision,
        )
        retrieval = RetrievalSpec(llm=llm, top_k=6)
        pros = ["Vision-assisted extraction for charts and scans"]
        cons = ["Slower ingestion", "Vision model cost"]
    else:
        ingestion = IngestionSpec(embedding=embedding)
        retrieval = RetrievalSpec(llm=llm, top_k=4)
        pros = ["Balanced default pipeline", "Lower complexity"]
        cons = ["Weaker citations than citation architecture"]

    return PipelineSpec(
        id=f"{architecture.value}-default",
        architecture=architecture,
        ingestion=ingestion,
        retrieval=retrieval,
        pros=pros,
        cons=cons,
    )


def build_candidate_specs(
    requirements: Requirements,
    architectures: list[RagArchitecture],
    embedding: AdapterRef,
    llm_mid: AdapterRef,
    llm_flagship: AdapterRef | None,
    vision: AdapterRef | None,
) -> list[PipelineSpec]:
    """Rule-based spec builder; customizer LLM can refine later."""
    specs: list[PipelineSpec] = []
    llm = llm_mid
    if requirements.budget_monthly_usd and requirements.budget_monthly_usd > 3000:
        llm = llm_flagship or llm_mid

    for arch in architectures:
        spec = _default_specs_for_architecture(arch, embedding, llm, vision)
        specs.append(spec)

    # Retrieval variants for backfill (D4) when fewer than 3 architectures
    if len(specs) < 3 and llm_flagship and llm_flagship.model != llm_mid.model:
        naive = _default_specs_for_architecture(
            RagArchitecture.NAIVE, embedding, llm_flagship, vision
        )
        naive.id = "naive-flagship-llm"
        naive.cons = ["Higher per-query cost"]
        specs.append(naive)

    return specs[:5]


def refine_with_llm(
    requirements: Requirements,
    specs: list[PipelineSpec],
    provider: str | None = None,
    model: str | None = None,
) -> list[PipelineSpec]:
    """Optional LLM pass for rationale/pros/cons; falls back on failure."""
    rules = _load_rules()
    custom = rules.get("customizer", {})
    prov = provider or custom.get("default_provider", "gemini")
    mdl = model or custom.get("default_model", "gemini-2.0-flash")

    prompt = (
        "Given requirements and pipeline specs as JSON, return the same specs array "
        "with updated rationale, pros, cons per spec. Return ONLY valid JSON array.\n"
        f"Requirements: {requirements.model_dump_json()}\n"
        f"Specs: {json.dumps([s.model_dump() for s in specs])}"
    )
    try:
        llm = create_llm(prov, model=mdl)
        raw = llm.generate(prompt)
        data = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        return [PipelineSpec.model_validate(item) for item in data]
    except Exception as exc:
        logger.warning("Customizer LLM failed, using rule-based specs: %s", exc)
        for spec in specs:
            if not spec.rationale:
                spec.rationale = (
                    f"Selected {spec.architecture.value} for: {requirements.use_case[:120]}"
                )
        return specs
