"""Persist recommendation run artifacts for local inspection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from orchestration.models import PipelineBlueprint, Requirements, ScoredCandidate


def write_experiment_artifact(
    *,
    run_id: UUID,
    requirements: Requirements,
    scored: list[ScoredCandidate],
    blueprint: PipelineBlueprint,
) -> None:
    state_dir = Path("state/experiments")
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": str(run_id),
        "requirements": requirements.model_dump(mode="json"),
        "scored_candidates": [s.model_dump(mode="json") for s in scored],
        "blueprint": blueprint.model_dump(mode="json"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    (state_dir / f"{run_id}.json").write_text(json.dumps(payload, indent=2))
