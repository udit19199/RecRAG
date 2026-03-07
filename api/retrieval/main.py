import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    # Async evaluation job id (if eval requested async)
    eval_job_id: str | None = None
    # Synchronous eval scores (if eval requested sync)
    eval: dict[str, float] | None = None


class EvaluateRequest(BaseModel):
    query: str
    contexts: list[str]
    response: str
    ground_truth: str | None = None


class EvaluateResponse(BaseModel):
    scores: dict[str, float]


class EvalJobStatus(BaseModel):
    id: str
    status: str
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


# Simple in-memory job store for evaluation jobs. For production replace with
# Redis, database or task queue to persist across restarts.
_eval_jobs: dict[str, dict] = {}


def _run_eval_job(
    job_id: str,
    query: str,
    contexts: list[str],
    response: str,
    ground_truth: str | None,
) -> None:
    """Background worker to run RAGAS evaluation and store result in _eval_jobs."""
    try:
        from evaluation.ragas_eval import get_evaluator

        evaluator = get_evaluator()
        scores = evaluator.evaluate_query(
            query, contexts, response, ground_truth=ground_truth
        )
        _eval_jobs[job_id] = {"status": "complete", "scores": scores}
    except Exception as exc:
        _eval_jobs[job_id] = {"status": "error", "error": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest, background_tasks: BackgroundTasks, eval: str | None = None
) -> QueryResponse:
    """Query the retrieval pipeline.

    Optional query param `eval` may be:
      - None or 'false' (default): no evaluation
      - 'sync': run evaluation synchronously and return scores inline
      - 'async': schedule evaluation in background and return `eval_job_id`
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        pipeline = get_pipeline()
        result = pipeline.query(request.query)

        context = [
            ContextItem(text=doc.text, source=doc.source, distance=doc.distance)
            for doc in result["context"]
        ]

        # If evaluation not requested, return normal response
        if not eval or eval.lower() in ("false", "0"):
            return QueryResponse(response=result["response"], context=context)

        # Prepare contexts as plain strings for evaluator
        contexts_texts = [doc.text for doc in result["context"]]

        # Synchronous evaluation (blocking)
        if eval.lower() == "sync":
            try:
                from evaluation.ragas_eval import get_evaluator

                evaluator = get_evaluator()
                scores = evaluator.evaluate_query(
                    request.query, contexts_texts, result["response"]
                )
                return QueryResponse(
                    response=result["response"], context=context, eval=scores
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

        # Asynchronous evaluation: schedule background task and return job id
        if eval.lower() == "async":
            import uuid

            job_id = str(uuid.uuid4())
            _eval_jobs[job_id] = {"status": "pending"}
            background_tasks.add_task(
                _run_eval_job,
                job_id,
                request.query,
                contexts_texts,
                result["response"],
                None,
            )
            return QueryResponse(
                response=result["response"], context=context, eval_job_id=job_id
            )

        raise HTTPException(status_code=400, detail=f"Unknown eval mode: {eval}")
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


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_once(req: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a single query/response pair synchronously.

    This endpoint is primarily for frontend dev and debugging; it calls the
    same RAGAS evaluator used by background jobs.
    """
    try:
        from evaluation.ragas_eval import get_evaluator

        evaluator = get_evaluator()
        scores = evaluator.evaluate_query(
            req.query, req.contexts, req.response, ground_truth=req.ground_truth
        )
        return EvaluateResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


@app.get("/evaluate/{job_id}", response_model=EvalJobStatus)
async def eval_status(job_id: str) -> EvalJobStatus:
    """Get status of an async evaluation job."""
    job = _eval_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    status = job.get("status", "pending")
    if status == "complete":
        return EvalJobStatus(id=job_id, status=status, scores=job.get("scores"))
    if status == "error":
        return EvalJobStatus(id=job_id, status=status, error=job.get("error"))
    return EvalJobStatus(id=job_id, status=status)
