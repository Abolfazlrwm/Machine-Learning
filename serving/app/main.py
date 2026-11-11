"""FastAPI application factory: routing, auth, CORS, error model, health."""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from . import __version__
from .exceptions import APIError
from .metrics import metrics as default_metrics
from .routers import rag, sms, stock
from .schemas import HealthResponse
from .services import ServiceRegistry

logger = logging.getLogger("serving")
logging.basicConfig(level=os.environ.get("SERVING_LOG_LEVEL", "INFO"))


def _verify_api_key(api_key: str | None):
    """Build the X-API-Key dependency (only when a key is configured)."""

    async def verify(x_api_key: str | None = Header(default=None)) -> None:
        if api_key and x_api_key != api_key:
            raise HTTPException(
                status_code=401,
                detail={"code": "UNAUTHORIZED",
                        "message": "missing or invalid X-API-Key header"},
            )

    return verify


def _route_template(request: Request) -> str:
    """Version-stable route template for metric labels.

    FastAPI >= 0.13x exposes the *router-relative* path on
    ``scope["route"]`` (``/rag/ask``) while older versions expose the
    mounted one (``/v1/rag/ask``). Re-attach the URL prefix so metric
    labels do not depend on the installed FastAPI version.
    """
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if not route_path:
        return request.url.path
    url_path = request.url.path
    if url_path != route_path and url_path.endswith(route_path):
        return url_path[: len(url_path) - len(route_path)] + route_path
    return route_path


def create_app(
    registry: ServiceRegistry | None = None,
    *,
    api_key: str | None = None,
    metrics=None,
) -> FastAPI:
    """Build the app. ``registry``/``api_key``/``metrics`` are injectable."""
    registry = registry or ServiceRegistry()
    api_key = api_key if api_key is not None else os.environ.get("SERVING_API_KEY") or None
    metrics_reg = metrics if metrics is not None else default_metrics

    app = FastAPI(
        title="ML Portfolio API",
        version=__version__,
        summary="One serving layer for the Persian RAG assistant, the SMS "
                "spam classifier and the stock movement models.",
        description=(
            "Endpoints under `/v1` expose the three model families of this "
            "portfolio. Services whose local artifacts are missing answer "
            "`503` with a machine-readable error code and a fix-it hint."
        ),
    )
    app.state.registry = registry
    app.state.metrics = metrics_reg
    app.state.started_at = time.monotonic()

    # ---- middleware -----------------------------------------------------
    cors_origins = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        """Request ID + latency header + Prometheus metrics in one place."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        took_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{took_ms:.1f}"
        # Route template ("/v1/rag/ask") keeps label cardinality bounded;
        # unmatched paths fall back to the raw path.
        route = _route_template(request)
        if metrics_reg is not None:
            metrics_reg.inc(
                "http_requests",
                method=request.method,
                path=route,
                status=str(response.status_code),
            )
            metrics_reg.observe(
                "http_request_duration_seconds",
                took_ms / 1000.0,
                method=request.method,
                path=route,
            )
        logger.info(
            "[%s] %s %s -> %s (%.1f ms)",
            request_id[:8], request.method, request.url.path,
            response.status_code, took_ms,
        )
        return response

    # ---- error model ----------------------------------------------------
    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError) -> JSONResponse:
        payload = {"error": {"code": exc.code, "message": exc.message}}
        if exc.hint:
            payload["error"]["hint"] = exc.hint
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        body = {"error": {"code": detail.get("code", "HTTP_ERROR"), **{
            k: v for k, v in detail.items() if k != "code"
        }}}
        return JSONResponse(status_code=exc.status_code, content=body)

    # ---- routes ---------------------------------------------------------
    guard = [Depends(_verify_api_key(api_key))] if api_key else None
    app.include_router(rag.router, prefix="/v1", dependencies=guard)
    app.include_router(sms.router, prefix="/v1", dependencies=guard)
    app.include_router(stock.router, prefix="/v1", dependencies=guard)

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {
            "name": "ML Portfolio API",
            "version": __version__,
            "docs": "/docs",
            "health": "/v1/health",
            "endpoints": [
                "/v1/rag/ask", "/v1/sms/classify", "/v1/stock/signal",
                "/v1/metrics",
            ],
        }

    @app.get("/v1/metrics", tags=["meta"])
    async def prometheus_metrics() -> Response:
        """Prometheus text exposition (v0.0.4) — rendered from scratch."""
        return Response(
            content=metrics_reg.render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        services = registry.statuses()
        return HealthResponse(
            status="ok" if "unavailable" not in services.values() else "degraded",
            version=__version__,
            services=services,  # type: ignore[arg-type]
            uptime_seconds=round(time.monotonic() - app.state.started_at, 3),
        )

    return app


app = create_app()
