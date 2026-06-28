"""Collection naming including ingest policy (TC-2, D20)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from orchestration.models import PipelineSpec


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value.replace("/", "_").replace("-", "_"))


def ingest_policy_hash(spec: PipelineSpec) -> str:
    """Stable short hash for chunk policy + architecture + extraction mode."""
    payload = (
        f"{spec.architecture.value}|"
        f"{spec.ingestion.extraction_mode}|"
        f"{spec.ingestion.chunk_size}|"
        f"{spec.ingestion.chunk_overlap}|"
        f"{spec.ingestion.metadata_level}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def get_collection_name_for_spec(
    config: dict[str, Any],
    spec: PipelineSpec,
    run_id: str | None = None,
) -> str:
    """Build Milvus collection name unique per candidate index (D20)."""
    prefix = config.get("storage", {}).get("collection_prefix", "recrag_")
    embed_id = _slug(spec.ingestion.embedding.model)
    arch = spec.architecture.value
    policy = ingest_policy_hash(spec)
    vision = spec.ingestion.vision
    if vision and spec.ingestion.extraction_mode == "vision_assisted":
        vision_id = _slug(vision.model)
        base = f"{prefix}{arch}_{vision_id}_{embed_id}_{policy}"
    else:
        base = f"{prefix}{arch}_text_{embed_id}_{policy}"
    if run_id:
        return f"{base}_{run_id[:8]}"
    return base
