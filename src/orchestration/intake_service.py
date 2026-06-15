"""Intake session handling (D23)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy.orm import Session

from adapters import create_llm
from orchestration.db.models import IntakeSessionRow
from orchestration.models import Requirements


REQUIRED_FIELDS = ("use_case", "document_modality", "audience")


def _merge_partial(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = {**current, **{k: v for k, v in update.items() if v is not None}}
    return merged


def extract_from_text(text: str) -> dict[str, Any]:
    """Lightweight extraction; LLM refine optional."""
    partial: dict[str, Any] = {}
    lower = text.lower()
    if "citation" in lower or "compliance" in lower:
        partial["citations_required"] = True
    if "image" in lower or "scan" in lower or "chart" in lower:
        partial["document_modality"] = "image_heavy"
    elif "video" in lower:
        partial["document_modality"] = "video"
    else:
        partial.setdefault("document_modality", "text_heavy")
    partial["use_case"] = text.strip()[:2000]
    return partial


def llm_extract(text: str, provider: str = "gemini", model: str = "gemini-2.0-flash") -> dict[str, Any]:
    prompt = (
        "Extract Requirements fields as JSON with keys: use_case, audience, "
        "document_modality (text_heavy|scanned|image_heavy|video|mixed), "
        "budget_monthly_usd (number or null), citations_required (bool), user_queries (array). "
        "Return ONLY JSON.\n\n" + text
    )
    try:
        llm = create_llm(provider, model=model)
        raw = llm.generate(prompt)
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
    except Exception:
        return extract_from_text(text)


class IntakeService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_session(self, workspace_id: uuid.UUID) -> IntakeSessionRow:
        row = IntakeSessionRow(workspace_id=workspace_id, partial_requirements={}, messages=[])
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def get_session(self, session_id: uuid.UUID) -> IntakeSessionRow | None:
        return self.db.get(IntakeSessionRow, session_id)

    def chat_turn(
        self,
        workspace_id: uuid.UUID,
        message: str,
        session_id: uuid.UUID | None = None,
    ) -> tuple[IntakeSessionRow, str]:
        if session_id:
            row = self.get_session(session_id)
            if row is None or row.workspace_id != workspace_id:
                raise ValueError("Invalid intake session")
        else:
            row = self.create_session(workspace_id)

        messages = list(row.messages or [])
        messages.append({"role": "user", "content": message})
        update = llm_extract(message)
        partial = _merge_partial(row.partial_requirements or {}, update)
        row.partial_requirements = partial
        row.messages = messages

        missing = [f for f in REQUIRED_FIELDS if not partial.get(f)]
        if missing:
            assistant = (
                f"Thanks. Please tell me more about: {', '.join(missing)}."
            )
            row.complete = False
        else:
            assistant = "Requirements complete. You can start a recommendation run."
            row.complete = True

        messages.append({"role": "assistant", "content": assistant})
        row.messages = messages
        self.db.commit()
        self.db.refresh(row)
        return row, assistant

    def to_requirements(self, row: IntakeSessionRow) -> Requirements:
        return Requirements.model_validate(row.partial_requirements)
