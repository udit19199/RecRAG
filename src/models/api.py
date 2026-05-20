from enum import StrEnum

from pydantic import BaseModel


class ExtractionMode(StrEnum):
    """PDF extraction mode for ingestion."""

    TEXT_ONLY = "text_only"
    VISION_ASSISTED = "vision_assisted"


class AdapterConfig(BaseModel):
    provider: str
    model: str


class VisionConfig(BaseModel):
    """Configuration for vision-assisted extraction."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None


# Ingestion models
class FileListResponse(BaseModel):
    files: list[str]


class StatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int | None = None
    error_message: str | None = None
    extraction_mode: str | None = None


class UploadRequest(BaseModel):
    """Request body for upload endpoint (used with form data)."""

    extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY
    vision_provider: str | None = None
    vision_model: str | None = None


class UploadResponse(BaseModel):
    success: bool
    files_uploaded: int
    message: str
    extraction_mode: str


class EmbeddingConfigPatch(BaseModel):
    embedding: AdapterConfig


class IndexStatusRequest(BaseModel):
    embedding: AdapterConfig | None = None
    vision: AdapterConfig | None = None


class IndexStatusResponse(BaseModel):
    has_documents: bool


class TargetedIngestRequest(BaseModel):
    extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY
    vision_provider: str | None = None
    vision_model: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None


class ReindexRequest(BaseModel):
    """Request body for reindex endpoint."""

    extraction_mode: ExtractionMode = ExtractionMode.TEXT_ONLY
    vision_provider: str | None = None
    vision_model: str | None = None


class ReindexResponse(BaseModel):
    started: bool
    message: str
    extraction_mode: str


# Retrieval models
class QueryRequest(BaseModel):
    query: str
    llm: AdapterConfig | None = None
    embedding: AdapterConfig | None = None
    vision: AdapterConfig | None = None


class ContextItem(BaseModel):
    text: str
    source: str
    distance: float


class QueryResponse(BaseModel):
    response: str
    context: list[ContextItem]
    eval_job_id: str | None = None


class EvalJobStatus(BaseModel):
    id: str
    status: str
    query: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    scores: dict[str, float] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    pipeline_loaded: bool
    has_documents: bool = False
    error_message: str | None = None


class ProviderInfo(BaseModel):
    available: bool
    models: list[str]
    reason: str | None = None


class ProvidersResponse(BaseModel):
    embedders: dict[str, ProviderInfo]
    llms: dict[str, ProviderInfo]
    vision: dict[str, ProviderInfo] | None = None


class ConfigResponse(BaseModel):
    embedding: AdapterConfig
    llm: AdapterConfig


class SetConfigResponse(BaseModel):
    applied: bool
    requires_reindex: bool
    embedding: AdapterConfig
    llm: AdapterConfig


class ConfigUpdateRequest(BaseModel):
    embedding: AdapterConfig | None = None
    llm: AdapterConfig | None = None
