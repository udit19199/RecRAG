"""Simple shared-secret JWT auth for internal research (replaces Clerk v1)."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from fastapi import Header, HTTPException, status
from pydantic import BaseModel

AUTH_USERNAME = os.environ.get("RECRAG_AUTH_USERNAME", "research")
AUTH_PASSWORD = os.environ.get("RECRAG_AUTH_PASSWORD", "research")


def _auth_secret() -> str:
    return os.environ.get("RECRAG_AUTH_SECRET", "").strip()


def auth_enabled() -> bool:
    return bool(_auth_secret())


@dataclass
class AuthContext:
    workspace_key: str
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str


def create_access_token(username: str) -> str:
    import jwt

    if not auth_enabled():
        msg = "RECRAG_AUTH_SECRET is not configured"
        raise RuntimeError(msg)
    return jwt.encode({"sub": username}, _auth_secret(), algorithm="HS256")


def verify_access_token(token: str) -> str:
    import jwt

    if not auth_enabled():
        msg = "RECRAG_AUTH_SECRET is not configured"
        raise RuntimeError(msg)
    payload = jwt.decode(token, _auth_secret(), algorithms=["HS256"])
    username = payload.get("sub")
    if not username or not isinstance(username, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token",
        )
    return username


def verify_credentials(username: str, password: str) -> bool:
    return username == AUTH_USERNAME and password == AUTH_PASSWORD


async def get_auth_context(
    authorization: str | None = Header(default=None),
) -> AuthContext:
    """Verify Bearer JWT when auth is enabled; open dev workspace otherwise."""
    if auth_enabled():
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Sign in required",
            )
        token = authorization.removeprefix("Bearer ").strip()
        try:
            username = verify_access_token(token)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
            ) from exc
        return AuthContext(workspace_key=username, username=username)

    return AuthContext(workspace_key="dev", username="dev")


def workspace_id_for_key(workspace_key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"recrag-user:{workspace_key}")
