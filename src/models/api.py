from pydantic import BaseModel


class AdapterConfig(BaseModel):
    provider: str
    model: str


# Ingestion models
class StatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    files_processed: int | None = None
    error_message: str | None = None


class UploadResponse(BaseModel):
    success: bool
    files_uploaded: int
    message: str


class EmbeddingConfigPatch(BaseModel):
    embedding: AdapterConfig


class ReindexResponse(BaseModel):
    started: bool
    message: str


# Retrieval models
class QueryRequest(BaseModel):
    query: str


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


class ProviderInfo(BaseModel):
    available: bool
    models: list[str]
    reason: str | None = None


class ProvidersResponse(BaseModel):
    embedders: dict[str, ProviderInfo]
    llms: dict[str, ProviderInfo]


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
