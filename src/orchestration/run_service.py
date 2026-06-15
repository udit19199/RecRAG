"""Recommendation run lifecycle (D2, D12)."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from benchmark.runner import run_benchmark
from config import find_config_path, load_config
from orchestration.clients import (
    fetch_providers,
    index_candidates_sequential,
    list_corpus_files,
)
from orchestration.db.models import RecommendationRunRow, WorkspaceRow
from orchestration.engine import OrchestrationEngine
from orchestration.models import (
    BenchmarkSummary,
    ConfidenceLevel,
    PipelineBlueprint,
    PipelineCandidate,
    Requirements,
    RetentionChoice,
    RetentionDuration,
    RunStatus,
    ScoredCandidate,
)
from orchestration.resolver import build_provider_model_map
from research.artifacts import write_experiment_artifact

logger = logging.getLogger(__name__)


class RunService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.retrieval_url = os.environ.get(
            "RETRIEVAL_API_URL", "http://127.0.0.1:8000"
        )
        self.ingestion_url = os.environ.get(
            "INGESTION_API_URL", "http://127.0.0.1:8001"
        )
        config_path = find_config_path()
        self.config = load_config(config_path)
        self.engine = OrchestrationEngine(self.config)

    def create_run(self, workspace_id: uuid.UUID, requirements: Requirements) -> uuid.UUID:
        run_id = uuid.uuid4()
        row = RecommendationRunRow(
            id=run_id,
            workspace_id=workspace_id,
            status=RunStatus.PENDING.value,
            requirements=requirements.model_dump(mode="json"),
        )
        self.db.add(row)
        self.db.commit()
        return run_id

    async def execute_run(self, run_id: uuid.UUID) -> None:
        row = self.db.get(RecommendationRunRow, run_id)
        if row is None:
            return
        requirements = Requirements.model_validate(row.requirements)
        row.status = RunStatus.SHORTLISTING.value
        self.db.commit()

        try:
            providers_raw = await fetch_providers(self.retrieval_url)
            provider_models = build_provider_model_map(
                providers_raw.get("embedders", {}),
                providers_raw.get("llms", {}),
                providers_raw.get("vision"),
            )
            candidates = self.engine.shortlist(
                requirements,
                provider_models,
                run_id=str(run_id),
            )
            row.candidates = [c.model_dump(mode="json") for c in candidates]
            row.preliminary = self.engine.preliminary(
                requirements, candidates
            ).model_dump(mode="json")
            self.db.commit()

            files = await list_corpus_files(self.ingestion_url)
            if not files:
                row.status = RunStatus.COMPLETE.value
                self.db.commit()
                return

            row.status = RunStatus.INDEXING.value
            self.db.commit()
            indexed = await index_candidates_sequential(
                self.ingestion_url, candidates
            )
            row.collection_names = [c.spec.collection_name for c in indexed]
            if len(indexed) < 1:
                row.status = RunStatus.FAILED.value
                row.error_message = "No candidates indexed successfully"
                self.db.commit()
                return

            row.status = RunStatus.BENCHMARKING.value
            self.db.commit()
            weights = self.engine.metric_weights(requirements)
            scored = await run_benchmark(
                requirements,
                indexed,
                weights,
                self.retrieval_url,
                os.environ.get("REC_RAG_API_KEY"),
            )
            if not scored:
                row.status = RunStatus.FAILED.value
                row.error_message = "Benchmark produced no scores"
                self.db.commit()
                return

            row.scored_candidates = [s.model_dump(mode="json") for s in scored]
            winner = scored[0]
            blueprint = PipelineBlueprint(
                run_id=run_id,
                workspace_id=row.workspace_id,
                architecture=winner.spec.architecture,
                ingestion=winner.spec.ingestion,
                retrieval=winner.spec.retrieval,
                rationale=winner.spec.rationale,
                pros=winner.spec.pros,
                cons=winner.spec.cons,
                estimated_monthly_usd=winner.spec.estimated_monthly_usd,
                benchmark_summary=winner.benchmark_summary,
            )
            row.blueprint = blueprint.model_dump(mode="json")
            row.status = RunStatus.COMPLETE.value
            self.db.commit()

            write_experiment_artifact(
                run_id=run_id,
                requirements=requirements,
                scored=scored,
                blueprint=blueprint,
            )
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            row.status = RunStatus.FAILED.value
            row.error_message = str(exc)
            self.db.commit()

    def apply_retention(
        self,
        run_id: uuid.UUID,
        choice: RetentionChoice,
        duration: RetentionDuration | None = None,
    ) -> None:
        row = self.db.get(RecommendationRunRow, run_id)
        if row is None:
            return
        row.retention_choice = choice.value
        if choice == RetentionChoice.LATER and duration:
            hours = {"1h": 1, "24h": 24, "7d": 24 * 7}[duration.value]
            row.retention_until = datetime.utcnow() + timedelta(hours=hours)
        elif choice == RetentionChoice.NO:
            row.retention_until = datetime.utcnow()
        self.db.commit()

    async def _teardown_collections(self, names: list[str]) -> None:
        # D22 — async teardown; Milvus drop deferred to future admin task
        logger.info("Scheduled teardown for collections: %s", names)


def get_or_create_workspace(db: Session, clerk_org_id: str, name: str | None = None) -> WorkspaceRow:
    row = db.query(WorkspaceRow).filter_by(clerk_org_id=clerk_org_id).one_or_none()
    if row:
        return row
    row = WorkspaceRow(clerk_org_id=clerk_org_id, name=name)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
