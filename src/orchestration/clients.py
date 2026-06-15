"""HTTP clients for ingestion and retrieval (D28)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from models.api import ExtractionMode
from orchestration.models import PipelineCandidate

logger = logging.getLogger(__name__)


def _api_key() -> str | None:
    return os.environ.get("REC_RAG_API_KEY") or None


def _headers() -> dict[str, str]:
    key = _api_key()
    return {"RecRAG-API-Key": key} if key else {}


async def fetch_providers(retrieval_url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=retrieval_url, timeout=30.0) as client:
        resp = await client.get("/providers", headers=_headers())
        resp.raise_for_status()
        return resp.json()


async def list_corpus_files(ingestion_url: str) -> list[str]:
    async with httpx.AsyncClient(base_url=ingestion_url, timeout=30.0) as client:
        resp = await client.get("/files", headers=_headers())
        resp.raise_for_status()
        return resp.json().get("files", [])


async def trigger_targeted_ingest(
    ingestion_url: str,
    candidate: PipelineCandidate,
) -> None:
    spec = candidate.spec
    body: dict[str, Any] = {
        "extraction_mode": spec.ingestion.extraction_mode,
        "embedding_provider": spec.ingestion.embedding.provider,
        "embedding_model": spec.ingestion.embedding.model,
        "collection_name": spec.collection_name,
        "chunk_size": spec.ingestion.chunk_size,
        "chunk_overlap": spec.ingestion.chunk_overlap,
    }
    if spec.ingestion.vision:
        body["vision_provider"] = spec.ingestion.vision.provider
        body["vision_model"] = spec.ingestion.vision.model
    if spec.ingestion.extraction_mode == "vision_assisted":
        body["extraction_mode"] = ExtractionMode.VISION_ASSISTED.value

    async with httpx.AsyncClient(base_url=ingestion_url, timeout=30.0) as client:
        resp = await client.post("/ingest/target", json=body, headers=_headers())
        resp.raise_for_status()


async def wait_for_ingestion(ingestion_url: str, timeout_s: float = 600.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    async with httpx.AsyncClient(base_url=ingestion_url, timeout=30.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            resp = await client.get("/status", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "complete":
                return True
            if status == "error":
                logger.error("Ingestion error: %s", data.get("error_message"))
                return False
            await asyncio.sleep(2.0)
    return False


async def index_candidates_sequential(
    ingestion_url: str,
    candidates: list[PipelineCandidate],
) -> list[PipelineCandidate]:
    """BM-5: sequential due to ingestion single-job lock."""
    indexed: list[PipelineCandidate] = []
    for candidate in candidates:
        try:
            await trigger_targeted_ingest(ingestion_url, candidate)
            ok = await wait_for_ingestion(ingestion_url)
            if ok:
                indexed.append(candidate)
        except Exception as exc:
            logger.warning("Index failed for %s: %s", candidate.spec.id, exc)
    return indexed
