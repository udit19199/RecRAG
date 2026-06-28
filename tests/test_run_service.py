"""Run lifecycle — corpus optional fast path (D12)."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from orchestration.customizer import build_candidate_specs
from orchestration.db.models import RecommendationRunRow
from orchestration.engine import OrchestrationEngine
from orchestration.models import (
    AdapterRef,
    PipelineCandidate,
    RagArchitecture,
    Requirements,
    RunStatus,
)
from orchestration.run_service import NO_CORPUS_NOTE, RunService


@pytest.fixture
def db_session() -> MagicMock:
    session = MagicMock()
    session.commit = MagicMock()
    return session


def _sample_candidate() -> PipelineCandidate:
    spec = build_candidate_specs(
        Requirements(use_case="test"),
        [RagArchitecture.NAIVE],
        AdapterRef(provider="gemini", model="gemini-embedding-001"),
        AdapterRef(provider="gemini", model="gemini-2.0-flash"),
        None,
        None,
    )[0]
    return PipelineCandidate(spec=spec, feasibility_score=0.9)


def test_complete_preliminary_without_corpus_note(db_session: MagicMock) -> None:
    row = RecommendationRunRow(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        status=RunStatus.SHORTLISTING.value,
        requirements={"use_case": "policy search"},
    )
    service = RunService(db_session)
    candidates = [_sample_candidate()]
    requirements = Requirements(use_case="policy search")

    service._complete_preliminary(
        row, requirements, candidates, note=NO_CORPUS_NOTE
    )

    assert row.status == RunStatus.COMPLETE.value
    assert row.error_message is None
    assert row.preliminary is not None
    assert row.preliminary.get("note") == NO_CORPUS_NOTE


def test_engine_preliminary_with_note_field() -> None:
    engine = OrchestrationEngine({})
    prelim = engine.preliminary(
        Requirements(use_case="test"),
        [_sample_candidate()],
    ).model_copy(update={"note": "optional note"})
    assert prelim.note == "optional note"
