"""FastAPI application factory: routing, auth, CORS, error model, health."""
from __future__ import annotations

import logging
import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .exceptions import APIError
from .routers import rag
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


def create_app(
    registry: ServiceRegistry | None = None,
    *,
    api_key: str | None = None,
) -> FastAPI:
    """Build the app. ``registry``/``api_key`` are injectable for tests."""
    registry = registry or ServiceRegistry()
    api_key = api_key if api_key is not None else os.environ.get("SERVING_API_KEY") or None

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
    async def add_process_time_header(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        took_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-Ms"] = f"{took_ms:.1f}"
        logger.info("%s %s -> %s (%.1f ms)",
                    request.method, request.url.path,
                    response.status_code, took_ms)
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

    @app.get("/", tags=["meta"])
    async def root() -> dict:
        return {
            "name": "ML Portfolio API",
            "version": __version__,
            "docs": "/docs",
            "health": "/v1/health",
            "endpoints": ["/v1/rag/ask", "/v1/sms/classify", "/v1/stock/signal"],
        }

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
