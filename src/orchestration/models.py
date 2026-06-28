"""Domain models for the Orchestrator recommendation system.

TC-1: PipelineBlueprint JSON schema — see PipelineBlueprint and BenchmarkSummary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RagArchitecture(StrEnum):
    """Active architectures for v1 benchmark; graph/agentic deferred (D15)."""

    NAIVE = "naive"
    CITATION = "citation"
    MULTIMODAL = "multimodal"
    # GRAPH = "graph"      # D15 — uncomment when implemented
    # AGENTIC = "agentic"  # D15 — uncomment when implemented


class DocumentModality(StrEnum):
    TEXT_HEAVY = "text_heavy"
    SCANNED = "scanned"
    IMAGE_HEAVY = "image_heavy"
    VIDEO = "video"
    MIXED = "mixed"


class ModelTier(StrEnum):
    FLAGSHIP = "flagship"
    MID = "mid"
    ECONOMY = "economy"


class ConfidenceLevel(StrEnum):
    PRELIMINARY = "preliminary"
    MEASURED = "measured"


class RetentionChoice(StrEnum):
    YES = "yes"
    NO = "no"
    LATER = "later"


class RetentionDuration(StrEnum):
    H1 = "1h"
    H24 = "24h"
    D7 = "7d"


class AdapterRef(BaseModel):
    provider: str
    model: str


class UsageAssumptions(BaseModel):
    document_pages: int | None = None
    queries_per_day: int | None = None
    avg_query_tokens: int | None = None
    avg_answer_tokens: int | None = None


class Requirements(BaseModel):
    """Structured intake from wizard or chat (D5)."""

    use_case: str
    audience: str = ""
    audience_expertise: Literal["junior", "mixed", "expert"] = "mixed"
    expected_query_volume: Literal["low", "medium", "high"] = "medium"
    document_modality: DocumentModality = DocumentModality.TEXT_HEAVY
    budget_monthly_usd: float | None = None
    citations_required: bool = False
    compliance_required: bool = False
    latency_sensitive: bool = False
    languages: list[str] = Field(default_factory=list)
    data_sensitive: bool = False
    usage: UsageAssumptions = Field(default_factory=UsageAssumptions)
    user_queries: list[str] = Field(default_factory=list)


class IngestionSpec(BaseModel):
    extraction_mode: Literal["text_only", "vision_assisted"] = "text_only"
    chunk_size: int = 1024
    chunk_overlap: int = 50
    metadata_level: Literal["basic", "full"] = "basic"
    embedding: AdapterRef
    vision: AdapterRef | None = None


class RetrievalSpec(BaseModel):
    llm: AdapterRef
    top_k: int = 4
    max_context_tokens: int = 4096
    reranker_enabled: bool = False
    context_template: str | None = None


class PipelineSpec(BaseModel):
    """Runnable pipeline configuration (slots + resolved models)."""

    id: str
    architecture: RagArchitecture
    ingestion: IngestionSpec
    retrieval: RetrievalSpec
    collection_name: str = ""
    rationale: str = ""
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    estimated_monthly_usd: float | None = None


class PipelineCandidate(BaseModel):
    """Pre-benchmark shortlist entry."""

    spec: PipelineSpec
    feasibility_score: float = 0.0


class BenchmarkSummary(BaseModel):
    """Compact evidence block in export JSON (D14)."""

    confidence: ConfidenceLevel = ConfidenceLevel.MEASURED
    candidates_evaluated: int = 0
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    composite_score: float | None = None
    margin_over_runner_up: float | None = None


class PipelineBlueprint(BaseModel):
    """TC-1 — exportable JSON deliverable after deep benchmark (D13, D14)."""

    run_id: UUID
    workspace_id: UUID
    architecture: RagArchitecture
    ingestion: IngestionSpec
    retrieval: RetrievalSpec
    rationale: str
    pros: list[str]
    cons: list[str]
    estimated_monthly_usd: float | None = None
    benchmark_summary: BenchmarkSummary
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ScoredCandidate(BaseModel):
    spec: PipelineSpec
    benchmark_summary: BenchmarkSummary
    rank: int


class BenchmarkSuite(BaseModel):
    domain_queries: list[str] = Field(default_factory=list)
    user_queries: list[str] = Field(default_factory=list)
    architecture_probes: dict[str, list[str]] = Field(default_factory=dict)

    def all_queries(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for q in (
            self.domain_queries
            + self.user_queries
            + [q for probes in self.architecture_probes.values() for q in probes]
        ):
            if q.strip() and q not in seen:
                seen.add(q)
                ordered.append(q)
        return ordered


class PreliminaryRecommendation(BaseModel):
    """Fast path — UI only, not exportable (D12)."""

    confidence: ConfidenceLevel = ConfidenceLevel.PRELIMINARY
    architecture: RagArchitecture
    rationale: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    estimated_monthly_usd: float | None = None
    candidates_preview: list[PipelineCandidate] = Field(default_factory=list)
    note: str | None = None


class RunStatus(StrEnum):
    PENDING = "pending"
    SHORTLISTING = "shortlisting"
    INDEXING = "indexing"
    BENCHMARKING = "benchmarking"
    COMPLETE = "complete"
    FAILED = "failed"


class RetentionRequest(BaseModel):
    choice: RetentionChoice
    duration: RetentionDuration | None = None


# --- TC-7 Orchestrator API request/response models ---


class CreateRunRequest(BaseModel):
    requirements: Requirements


class CreateRunResponse(BaseModel):
    run_id: UUID
    status: RunStatus


class RunResponse(BaseModel):
    run_id: UUID
    workspace_id: UUID
    status: RunStatus
    requirements: Requirements
    preliminary: PreliminaryRecommendation | None = None
    blueprint: PipelineBlueprint | None = None
    candidates: list[ScoredCandidate] = Field(default_factory=list)
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class RunListItem(BaseModel):
    run_id: UUID
    status: RunStatus
    architecture: RagArchitecture | None = None
    created_at: datetime


class RunListResponse(BaseModel):
    runs: list[RunListItem]


class IntakeChatRequest(BaseModel):
    session_id: UUID | None = None
    message: str


class IntakeChatResponse(BaseModel):
    session_id: UUID
    requirements_partial: dict[str, Any]
    complete: bool
    assistant_message: str


class IntakeExtractRequest(BaseModel):
    text: str


class IntakeExtractResponse(BaseModel):
    requirements_partial: dict[str, Any]


class CustomizerConfig(BaseModel):
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
