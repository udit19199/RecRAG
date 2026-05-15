"""API key authentication middleware for RecRAG FastAPI services."""

import os

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

# Health and docs endpoints that don't require authentication
PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

API_KEY_HEADER = APIKeyHeader(name="RecRAG-API-Key", auto_error=False)


def get_api_key() -> str | None:
    """Return the configured API key from environment.
    
    Returns None if no API key is configured (authentication disabled).
    """
    return os.environ.get("REC_RAG_API_KEY")


async def verify_api_key(request: Request) -> None:
    """Verify the API key for a request.
    
    Raises HTTPException if authentication fails.
    Skips authentication for public paths or when no API key is configured.
    """
    api_key = get_api_key()
    
    # Authentication disabled if no key configured
    if not api_key:
        return
    
    # Skip authentication for public endpoints
    if request.url.path in PUBLIC_PATHS:
        return
    
    # Verify API key
    provided_key = request.headers.get("x-api-key")
    if not provided_key or provided_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
