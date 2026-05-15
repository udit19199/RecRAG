"""Prometheus metrics middleware for RecRAG FastAPI services."""

import time
from collections.abc import Callable, Awaitable
from typing import Any

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Metrics definitions
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

REQUEST_EXCEPTIONS = Counter(
    "http_request_exceptions_total",
    "Total HTTP request exceptions",
    ["method", "path"],
)


class MetricsMiddleware:
    """ASGI middleware that collects Prometheus metrics."""

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

        request = Request(scope, receive=receive)
        method = request.method
        path = request.url.path
        start_time = time.time()

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status = str(message.get("status", 500))
                duration = time.time() - start_time
                
                REQUEST_COUNT.labels(
                    method=method,
                    path=path,
                    status=status,
                ).inc()
                
                REQUEST_DURATION.labels(
                    method=method,
                    path=path,
                ).observe(duration)

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            REQUEST_EXCEPTIONS.labels(
                method=method,
                path=path,
            ).inc()
            raise


def metrics_endpoint() -> Response:
    """Return Prometheus metrics endpoint response."""
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
