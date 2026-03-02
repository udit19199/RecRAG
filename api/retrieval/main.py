import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import find_config_path, load_config
from pipelines import get_retrieval_pipeline
from adapters import create_embedder, create_llm
from stores import VectorStore
from pipelines.base import get_collection_name, get_milvus_uri
from providers import (
    OPENAI_EMBEDDING_MODELS,
    OPENAI_LLM_MODELS,
    _fetch_nim_models,
    _fetch_ollama_models,
    _fetch_openai_models,
)

# ── Pydantic models ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str


class ContextItem(BaseModel):
    text: str
    source: str
    distance: float


class QueryResponse(BaseModel):
    response: str
    context: list[ContextItem]


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


class AdapterConfig(BaseModel):
    provider: str
    model: str


class ConfigResponse(BaseModel):
    embedding: AdapterConfig
    llm: AdapterConfig


class ConfigPatch(BaseModel):
    embedding: AdapterConfig | None = None
    llm: AdapterConfig | None = None


class SetConfigResponse(BaseModel):
    applied: bool
    requires_reindex: bool
    embedding: AdapterConfig
    llm: AdapterConfig


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecRAG Retrieval API",
    description="API for querying documents using Retrieval-Augmented Generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy loaded) + lock for thread-safe mutation
_pipeline: Any = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> Any:
    """Return the shared RetrievalPipeline, building it on first call."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                try:
                    config_path = find_config_path()
                    _pipeline = get_retrieval_pipeline(config_path)
                except Exception as e:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to initialize pipeline: {str(e)}",
                    )
    return _pipeline


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query)

        context = [
            ContextItem(text=doc.text, source=doc.source, distance=doc.distance)
            for doc in result["context"]
        ]

        return QueryResponse(response=result["response"], context=context)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    pipeline_loaded = _pipeline is not None
    if not pipeline_loaded:
        try:
            get_pipeline()
            pipeline_loaded = True
        except Exception:
            pipeline_loaded = False

    return HealthResponse(
        status="healthy" if pipeline_loaded else "starting",
        service="retrieval-api",
        pipeline_loaded=pipeline_loaded,
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return the currently active embedding and LLM configuration."""
    if _pipeline is not None:
        p = _pipeline
        return ConfigResponse(
            embedding=AdapterConfig(
                provider=getattr(p.embedder, "provider", "unknown"),
                model=getattr(p.embedder, "model", "unknown"),
            ),
            llm=AdapterConfig(
                provider=getattr(p.llm, "provider", "unknown"),
                model=getattr(p.llm, "model", "unknown"),
            ),
        )

    try:
        config_path = find_config_path()
        config = load_config(config_path)
        embed_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        return ConfigResponse(
            embedding=AdapterConfig(
                provider=embed_cfg.get("provider", "openai"),
                model=embed_cfg.get("model", "text-embedding-3-small"),
            ),
            llm=AdapterConfig(
                provider=llm_cfg.get("provider", "openai"),
                model=llm_cfg.get("model", "gpt-4o-mini"),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config: {e}")


@app.get("/providers", response_model=ProvidersResponse)
async def get_providers() -> ProvidersResponse:
    """Return available providers and their models, fetched live where possible."""
    ollama_embed_url = "http://localhost:11434"
    ollama_llm_url = "http://localhost:11434"

    try:
        config_path = find_config_path()
        config = load_config(config_path)
        embed_cfg = config.get("embedding", {})
        llm_cfg = config.get("llm", {})
        if embed_cfg.get("provider") == "ollama" and embed_cfg.get("base_url"):
            ollama_embed_url = embed_cfg["base_url"]
        if llm_cfg.get("provider") == "ollama" and llm_cfg.get("base_url"):
            ollama_llm_url = llm_cfg["base_url"]
    except Exception:
        pass

    ollama_embed_models, ollama_llm_models = _fetch_ollama_models(ollama_embed_url)
    if ollama_embed_url != ollama_llm_url:
        _, extra_llm = _fetch_ollama_models(ollama_llm_url)
        ollama_llm_models = sorted(set(ollama_llm_models + extra_llm))
    ollama_available = bool(ollama_embed_models or ollama_llm_models)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key:
        openai_embed_models, openai_llm_models = _fetch_openai_models(openai_key)
        openai_available = True
        openai_reason = None
    else:
        openai_embed_models = OPENAI_EMBEDDING_MODELS
        openai_llm_models = OPENAI_LLM_MODELS
        openai_available = True
        openai_reason = "OPENAI_API_KEY not set — showing default models"

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if nvidia_key:
        nim_embed_models, nim_llm_models = _fetch_nim_models(nvidia_key)
        nim_available = True
        nim_reason = None
    else:
        nim_embed_models = []
        nim_llm_models = []
        nim_available = False
        nim_reason = "NVIDIA_API_KEY not set"

    return ProvidersResponse(
        embedders={
            "ollama": ProviderInfo(
                available=ollama_available,
                models=ollama_embed_models,
                reason=None if ollama_available else "Ollama not reachable",
            ),
            "openai": ProviderInfo(
                available=openai_available,
                models=openai_embed_models,
                reason=openai_reason,
            ),
            "nim": ProviderInfo(
                available=nim_available,
                models=nim_embed_models,
                reason=nim_reason,
            ),
        },
        llms={
            "ollama": ProviderInfo(
                available=ollama_available,
                models=ollama_llm_models,
                reason=None if ollama_available else "Ollama not reachable",
            ),
            "openai": ProviderInfo(
                available=openai_available,
                models=openai_llm_models,
                reason=openai_reason,
            ),
            "nim": ProviderInfo(
                available=nim_available,
                models=nim_llm_models,
                reason=nim_reason,
            ),
        },
    )


@app.post("/config", response_model=SetConfigResponse)
async def set_config(patch: ConfigPatch) -> SetConfigResponse:
    """Swap the LLM and/or embedding model at runtime.

    - LLM swap: instantaneous, no data loss.
    - Embedding swap: safe, but the new vector collection will be empty until
      re-ingestion is triggered via the ingestion API's POST /reindex endpoint.
    """
    global _pipeline

    pipeline = get_pipeline()

    try:
        config_path = find_config_path()
        config = load_config(config_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")

    requires_reindex = False

    with _pipeline_lock:
        if patch.llm is not None:
            llm_section = config.get("llm", {})
            kwargs: dict[str, Any] = {
                k: v for k, v in llm_section.items() if k not in ("provider", "model")
            }
            try:
                new_llm = create_llm(
                    patch.llm.provider, model=patch.llm.model, **kwargs
                )
                pipeline.llm = new_llm
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create LLM adapter: {e}",
                )

        if patch.embedding is not None:
            embed_section = config.get("embedding", {})
            embed_kwargs: dict[str, Any] = {
                k: v for k, v in embed_section.items() if k not in ("provider", "model")
            }
            try:
                new_embedder = create_embedder(
                    patch.embedding.provider,
                    model=patch.embedding.model,
                    **embed_kwargs,
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create embedder adapter: {e}",
                )

            new_collection = get_collection_name(config, new_embedder.model)
            uri = get_milvus_uri(config, config_path)
            try:
                new_vs = VectorStore(
                    dimension=new_embedder.dimension,
                    collection_name=new_collection,
                    uri=uri,
                    metric_type=config.get("storage", {}).get("metric_type", "L2"),
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialise vector store: {e}",
                )

            try:
                count = new_vs.count
                requires_reindex = count == 0
            except Exception:
                requires_reindex = True

            pipeline.embedder = new_embedder
            pipeline.vector_store = new_vs

    return SetConfigResponse(
        applied=True,
        requires_reindex=requires_reindex,
        embedding=AdapterConfig(
            provider=getattr(pipeline.embedder, "provider", "unknown"),
            model=getattr(pipeline.embedder, "model", "unknown"),
        ),
        llm=AdapterConfig(
            provider=getattr(pipeline.llm, "provider", "unknown"),
            model=getattr(pipeline.llm, "model", "unknown"),
        ),
    )
