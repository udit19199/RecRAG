"""Orchestrator API — recommendation workflows (TC-7, D28)."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config import find_config_path, get_frontend_origins, load_config
from orchestration.db.session import get_db, init_db
from orchestration.intake_service import IntakeService, extract_from_text
from orchestration.models import (
    CreateRunRequest,
    CreateRunResponse,
    IntakeChatRequest,
    IntakeChatResponse,
    IntakeExtractRequest,
    IntakeExtractResponse,
    PipelineBlueprint,
    PreliminaryRecommendation,
    Requirements,
    RetentionRequest,
    RunListItem,
    RunListResponse,
    RunResponse,
    RunStatus,
    ScoredCandidate,
)
from orchestration.run_service import RunService, get_or_create_workspace

from .auth import (
    AuthContext,
    LoginRequest,
    LoginResponse,
    auth_enabled,
    create_access_token,
    get_auth_context,
    verify_credentials,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="RecRAG Orchestrator API",
    description="Pipeline recommendation intake, benchmark orchestration, and export",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_frontend_origins(load_config(find_config_path())),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Decoupled research experiments (kept out of the recommendation flow).
from .experiments.router import router as experiments_router  # noqa: E402

app.include_router(experiments_router)


def _workspace_id(auth: AuthContext, db: Session) -> uuid.UUID:
    ws = get_or_create_workspace(db, auth.workspace_key, name=auth.username)
    return ws.id


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "orchestrator", "auth": auth_enabled()}


@app.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    if not auth_enabled():
        raise HTTPException(
            status_code=503,
            detail="Auth is disabled — set RECRAG_AUTH_SECRET to enable sign-in",
        )
    if not verify_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return LoginResponse(
        token=create_access_token(body.username),
        username=body.username,
    )


# --- TC-7: Intake routes ---


@app.post("/intake/extract", response_model=IntakeExtractResponse)
async def intake_extract(body: IntakeExtractRequest) -> IntakeExtractResponse:
    return IntakeExtractResponse(requirements_partial=extract_from_text(body.text))


@app.post("/intake/chat", response_model=IntakeChatResponse)
async def intake_chat(
    body: IntakeChatRequest,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> IntakeChatResponse:
    service = IntakeService(db)
    ws_id = _workspace_id(auth, db)
    try:
        row, assistant = service.chat_turn(ws_id, body.message, body.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IntakeChatResponse(
        session_id=row.id,
        requirements_partial=row.partial_requirements,
        complete=row.complete,
        assistant_message=assistant,
    )


# --- TC-7: Run routes ---


@app.post("/runs", response_model=CreateRunResponse)
async def create_run(
    body: CreateRunRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> CreateRunResponse:
    ws_id = _workspace_id(auth, db)
    service = RunService(db)
    run_id = service.create_run(ws_id, body.requirements)
    background_tasks.add_task(_execute_run_task, run_id)
    return CreateRunResponse(run_id=run_id, status=RunStatus.PENDING)


async def _execute_run_task(run_id: uuid.UUID) -> None:
    from orchestration.db.session import SessionLocal

    db = SessionLocal()
    try:
        service = RunService(db)
        await service.execute_run(run_id)
    finally:
        db.close()


@app.get("/runs", response_model=RunListResponse)
async def list_runs(
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> RunListResponse:
    from orchestration.db.models import RecommendationRunRow
    from orchestration.models import RagArchitecture

    ws_id = _workspace_id(auth, db)
    runs = (
        db.query(RecommendationRunRow)
        .filter_by(workspace_id=ws_id)
        .order_by(RecommendationRunRow.created_at.desc())
        .limit(50)
        .all()
    )
    items: list[RunListItem] = []
    for r in runs:
        arch: RagArchitecture | None = None
        if r.blueprint and r.blueprint.get("architecture"):
            arch = RagArchitecture(r.blueprint["architecture"])
        items.append(
            RunListItem(
                run_id=r.id,
                status=RunStatus(r.status),
                architecture=arch,
                created_at=r.created_at,
            )
        )
    return RunListResponse(runs=items)


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(
    run_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> RunResponse:
    from orchestration.db.models import RecommendationRunRow

    ws_id = _workspace_id(auth, db)
    row = db.get(RecommendationRunRow, run_id)
    if row is None or row.workspace_id != ws_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(
        run_id=row.id,
        workspace_id=row.workspace_id,
        status=RunStatus(row.status),
        requirements=Requirements.model_validate(row.requirements),
        preliminary=(
            PreliminaryRecommendation.model_validate(row.preliminary)
            if row.preliminary
            else None
        ),
        blueprint=(
            PipelineBlueprint.model_validate(row.blueprint) if row.blueprint else None
        ),
        candidates=[
            ScoredCandidate.model_validate(c) for c in (row.scored_candidates or [])
        ],
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@app.get("/runs/{run_id}/export")
async def export_blueprint(
    run_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> dict:
    """D12 — export only when benchmark complete."""
    from orchestration.db.models import RecommendationRunRow

    ws_id = _workspace_id(auth, db)
    row = db.get(RecommendationRunRow, run_id)
    if row is None or row.workspace_id != ws_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if not row.blueprint:
        raise HTTPException(
            status_code=400,
            detail="Blueprint not available — complete deep benchmark first",
        )
    return row.blueprint


@app.post("/runs/{run_id}/retention")
async def set_retention(
    run_id: uuid.UUID,
    body: RetentionRequest,
    background_tasks: BackgroundTasks,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    from orchestration.db.models import RecommendationRunRow

    ws_id = _workspace_id(auth, db)
    row = db.get(RecommendationRunRow, run_id)
    if row is None or row.workspace_id != ws_id:
        raise HTTPException(status_code=404, detail="Run not found")
    service = RunService(db)
    service.apply_retention(run_id, body.choice, body.duration)
    if body.choice.value == "no":
        names = list(row.collection_names or [])

        async def _teardown() -> None:
            await service._teardown_collections(names)

        background_tasks.add_task(_teardown)
    return {"status": "ok"}
