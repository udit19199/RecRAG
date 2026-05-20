"""Structured logging middleware for RecRAG FastAPI services."""

import json
import logging
import time
import uuid
from collections.abc import Callable, Awaitable
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)


class StructuredLoggingMiddleware:
    """ASGI middleware that adds structured JSON logging with request IDs."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Create request object for logging
        request = Request(scope, receive=receive)
        method = request.method
        path = request.url.path

        # Add request ID to scope for downstream access
        scope["state"] = scope.get("state", {})
        scope["state"]["request_id"] = request_id

        async def log_response() -> None:
            status_code = None
            response_body = b""

            async def send_wrapper(message: dict[str, Any]) -> None:
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message.get("status", 500)
                elif message["type"] == "http.response.body":
                    nonlocal response_body
                    response_body += message.get("body", b"")
                await send(message)

            await self.app(scope, receive, send_wrapper)

            # Log structured JSON
            duration_ms = (time.time() - start_time) * 1000
            log_entry = {
                "timestamp": time.time(),
                "request_id": request_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": round(duration_ms, 2),
                "level": "error" if status_code and status_code >= 500 else "info",
            }

            log_method = (
                logger.error if status_code and status_code >= 500 else logger.info
            )
            log_method(json.dumps(log_entry))

        await log_response()
