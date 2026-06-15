"""Tests for orchestrator username/password auth."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.orchestrator.auth import create_access_token, verify_access_token
from app.orchestrator.main import app


@pytest.fixture
def auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECRAG_AUTH_SECRET", "test-secret")
    monkeypatch.setenv("RECRAG_AUTH_USERNAME", "research")
    monkeypatch.setenv("RECRAG_AUTH_PASSWORD", "research")


def test_create_and_verify_token(auth_env: None) -> None:
    token = create_access_token("research")
    assert verify_access_token(token) == "research"


def test_login_and_runs_require_token(
    auth_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestration.db.session import init_db

    async def _noop_run(_run_id: object) -> None:
        return None

    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr("app.orchestrator.main._execute_run_task", _noop_run)
    init_db()
    client = TestClient(app)

    unauth = client.post("/runs", json={"requirements": {"use_case": "test"}})
    assert unauth.status_code == 401

    bad_login = client.post(
        "/auth/login",
        json={"username": "research", "password": "wrong"},
    )
    assert bad_login.status_code == 401

    login = client.post(
        "/auth/login",
        json={"username": "research", "password": "research"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    authed = client.post(
        "/runs",
        json={"requirements": {"use_case": "test"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert authed.status_code == 200
    assert "run_id" in authed.json()


def test_auth_disabled_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RECRAG_AUTH_SECRET", raising=False)
    from orchestration.db.session import init_db

    async def _noop_run(_run_id: object) -> None:
        return None

    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr("app.orchestrator.main._execute_run_task", _noop_run)
    init_db()
    client = TestClient(app)
    res = client.post("/runs", json={"requirements": {"use_case": "test"}})
    assert res.status_code == 200
