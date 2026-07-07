"""Research experiment artifacts (D9, phase 7 — archive before prod)."""

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

    generated_dir = Path("docs/research/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)
    md = _markdown_summary(run_id, requirements, scored, blueprint)
    (generated_dir / f"{run_id}.md").write_text(md)


def _markdown_summary(
    run_id: UUID,
    requirements: Requirements,
    scored: list[ScoredCandidate],
    blueprint: PipelineBlueprint,
) -> str:
    lines = [
        f"# Experiment {run_id}",
        "",
        f"**Use case:** {requirements.use_case}",
        f"**Winner:** {blueprint.architecture.value}",
        "",
        "## Ranked candidates",
        "",
    ]
    for s in scored:
        score = s.benchmark_summary.composite_score
        lines.append(f"- #{s.rank} `{s.spec.architecture.value}` — score {score}")
    lines.extend(["", "## Rationale", "", blueprint.rationale, ""])
    return "\n".join(lines)
