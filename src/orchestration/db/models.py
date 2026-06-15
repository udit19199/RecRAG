"""SQLAlchemy models for Orchestrator (D21, D27)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


# SQLite fallback uses generic JSON instead of JSONB
JsonType = JSON().with_variant(JSONB, "postgresql")


class WorkspaceRow(Base):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_org_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list[RecommendationRunRow]] = relationship(back_populates="workspace")
    intake_sessions: Mapped[list[IntakeSessionRow]] = relationship(back_populates="workspace")


class IntakeSessionRow(Base):
    __tablename__ = "intake_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), index=True
    )
    partial_requirements: Mapped[dict] = mapped_column(JsonType, default=dict)
    messages: Mapped[list] = mapped_column(JsonType, default=list)
    complete: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    workspace: Mapped[WorkspaceRow] = relationship(back_populates="intake_sessions")


class RecommendationRunRow(Base):
    __tablename__ = "recommendation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    requirements: Mapped[dict] = mapped_column(JsonType, default=dict)
    candidates: Mapped[list] = mapped_column(JsonType, default=list)
    scored_candidates: Mapped[list] = mapped_column(JsonType, default=list)
    preliminary: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    blueprint: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retention_choice: Mapped[str | None] = mapped_column(String(16), nullable=True)
    retention_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    collection_names: Mapped[list] = mapped_column(JsonType, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    workspace: Mapped[WorkspaceRow] = relationship(back_populates="runs")
