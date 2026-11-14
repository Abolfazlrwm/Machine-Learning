"""/v1/rag — grounded question answering over the Persian corpus.

Both JSON and SSE (``stream=true``) responses are supported. The pipeline
streams real LLM deltas when the provider supports it (OpenAI SSE, Ollama
NDJSON); backends without a streaming protocol fall back to word-chunked
events *inside* the pipeline, so the SSE contract is identical either way:
``sources`` first, then ``token`` events, then ``done``. Failures raised
mid-stream surface as an SSE ``error`` event instead of a dropped
connection — the status line is already committed by then.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..exceptions import APIError
from ..schemas import AskRequest, AskResponse, InjectionMeta, RAGMeta, SourceModel

router = APIRouter(prefix="/rag", tags=["rag"])


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _ask_event_stream(
    rag_service,
    question: str,
    top_k: int | None,
    request_id: str,
    metrics_reg,
) -> Iterator[str]:
    """Map pipeline events to SSE frames; failures become ``error`` events."""
    try:
        for event in rag_service.ask_stream(question, top_k=top_k):
            kind = event["type"]
            if kind == "sources":
                data: dict = {
                    "sources": event["sources"],
                    "retrieval_mode": event["retrieval_mode"],
                }
                if "guardrails" in event:
                    data["guardrails"] = event["guardrails"]
                    data["injection"] = event["injection"]
                yield _sse("sources", data)
            elif kind == "token":
                yield _sse("token", {"text": event["text"]})
            elif kind == "done":
                meta = event.get("meta") or {}
                # Business metrics on the wire: cache outcome + stage latency
                # (pipeline reports ms; histograms want seconds).
                if metrics_reg is not None:
                    metrics_reg.inc("rag_cache_requests", result=event["cache"])
                    for stage in ("retrieve", "generate"):
                        timing_ms = (meta.get("timings_ms") or {}).get(stage)
                        if timing_ms is not None:
                            metrics_reg.observe(
                                "rag_stage_duration_seconds",
                                timing_ms / 1000.0,
                                stage=stage,
                            )
                yield _sse("done", {
                    "took_ms": event["took_ms"],
                    "request_id": request_id,
                    "cache": event["cache"],
                    "meta": meta,
                })
    except APIError as exc:
        # Mid-stream failures can't change the committed 200 status, so the
        # error travels as an SSE event with the shared error contract.
        payload = {"code": exc.code, "message": exc.message}
        if exc.hint:
            payload["hint"] = exc.hint
        yield _sse("error", payload)


@router.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(payload: AskRequest, request: Request):
    registry = request.app.state.registry
    started = time.perf_counter()

    if payload.stream:
        request_id = getattr(request.state, "request_id", "-")
        return StreamingResponse(
            _ask_event_stream(
                registry.rag,
                payload.question,
                payload.top_k,
                request_id,
                request.app.state.metrics,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = registry.rag.ask(payload.question, top_k=payload.top_k)
    took_ms = int((time.perf_counter() - started) * 1000)

    meta_source = result.get("meta") or {}
    # RAG business metrics: cache effectiveness + per-stage latency. The
    # pipeline reports stage timings in ms; histograms want seconds. The
    # registry comes from app.state so tests can inject isolated ones.
    metrics_reg = request.app.state.metrics
    if metrics_reg is not None:
        metrics_reg.inc(
            "rag_cache_requests", result=meta_source.get("cache", "off")
        )
        for stage in ("retrieve", "generate"):
            if stage in (meta_source.get("timings_ms") or {}):
                metrics_reg.observe(
                    "rag_stage_duration_seconds",
                    meta_source["timings_ms"][stage] / 1000.0,
                    stage=stage,
                )
    query_info = meta_source.get("query") or {}
    request_id = getattr(request.state, "request_id", "-")

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceModel(**s) for s in result["sources"]],
        retrieval_mode=result["retrieval_mode"],
        took_ms=took_ms,
        meta=RAGMeta(
            request_id=request_id,
            cache=meta_source.get("cache", "off"),
            timings_ms=meta_source.get("timings_ms") or None,
            query_expansions=query_info.get("expansions") or None,
            mixed_script=query_info.get("mixed_script"),
            guardrails=meta_source.get("guardrails"),
            injection=(
                InjectionMeta(**meta_source["injection"])
                if meta_source.get("injection")
                else None
            ),
        ),
    )
