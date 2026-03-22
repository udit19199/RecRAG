import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import find_config_path, get_frontend_origins, load_config
from providers import (
    _fetch_nim_models,
    _fetch_ollama_models,
    _fetch_openai_models,
)
from services.retrieval import (
    is_pipeline_loaded,
    run_eval_job,
    run_query,
    swap_runtime_config,
)
from utils.eval_jobs import read_eval_jobs
from models.api import (
    AdapterConfig,
    ConfigResponse,
    ConfigUpdateRequest,
    ContextItem,
    EvalJobStatus,
    HealthResponse,
    ProviderInfo,
    ProvidersResponse,
    QueryRequest,
    QueryResponse,
    SetConfigResponse,
)

STORAGE_DIR = Path("storage")

_IS_PRODUCTION = os.getenv("ENVIRONMENT", "").lower() == "production"

app = FastAPI(
    title="RecRAG Retrieval API",
    description="API for querying documents using Retrieval-Augmented Generation",
    version="1.0.0",
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    openapi_url=None if _IS_PRODUCTION else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(load_config(find_config_path())),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, background_tasks: BackgroundTasks
) -> QueryResponse:
    """Query the retrieval pipeline. Evaluation is always performed asynchronously."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = run_query(request.query, STORAGE_DIR)
        context = [
            ContextItem(text=doc.text, source=doc.source, distance=doc.distance)
            for doc in result.context
        ]

        background_tasks.add_task(
            run_eval_job,
            STORAGE_DIR,
            result.eval_job_id,
            request.query,
            [doc.text for doc in result.context],
            result.response,
            None,
        )
        return QueryResponse(
            response=result.response, context=context, eval_job_id=result.eval_job_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if is_pipeline_loaded() else "starting",
        service="retrieval-api",
        pipeline_loaded=is_pipeline_loaded(),
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Return the currently active embedding and LLM configuration."""
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
        openai_available = bool(openai_embed_models or openai_llm_models)
        openai_reason = None
    else:
        openai_embed_models = []
        openai_llm_models = []
        openai_available = False
        openai_reason = "OPENAI_API_KEY not set"

    nvidia_key = os.environ.get("NVIDIA_API_KEY", "")
    if nvidia_key:
        nim_embed_models, nim_llm_models = _fetch_nim_models(nvidia_key)
        nim_available = bool(nim_embed_models or nim_llm_models)
        nim_reason = None if nim_available else "No NIM models available"
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
async def set_config(patch: ConfigUpdateRequest) -> SetConfigResponse:
    try:
        config_path = find_config_path()
        config = load_config(config_path)
        embedding_payload = (
            {"provider": patch.embedding.provider, "model": patch.embedding.model}
            if patch.embedding is not None
            else None
        )
        llm_payload = (
            {"provider": patch.llm.provider, "model": patch.llm.model}
            if patch.llm is not None
            else None
        )
        (
            embed_provider,
            embed_model,
            llm_provider,
            llm_model,
            requires_reindex,
        ) = swap_runtime_config(
            config,
            config_path,
            embedding=embedding_payload,
            llm=llm_payload,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {exc}")

    return SetConfigResponse(
        applied=True,
        requires_reindex=requires_reindex,
        embedding=AdapterConfig(provider=embed_provider, model=embed_model),
        llm=AdapterConfig(provider=llm_provider, model=llm_model),
    )


@app.get("/evaluate/{job_id}", response_model=EvalJobStatus)
async def eval_status(job_id: str) -> EvalJobStatus:
    """Get status of an async evaluation job."""
    job = read_eval_jobs(STORAGE_DIR).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    status = job.get("status", "pending")
    if status == "complete":
        return EvalJobStatus(
            id=job_id,
            status=status,
            query=job.get("query"),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
            scores=job.get("scores"),
        )
    if status == "error":
        return EvalJobStatus(
            id=job_id,
            status=status,
            query=job.get("query"),
            created_at=job.get("created_at"),
            updated_at=job.get("updated_at"),
            error=job.get("error"),
        )
    return EvalJobStatus(
        id=job_id,
        status=status,
        query=job.get("query"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )
