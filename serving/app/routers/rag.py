"""/v1/rag — grounded question answering over the Persian corpus.

Both JSON and SSE (``stream=true``) responses are supported. The backing
LLM providers are non-streaming, so SSE delivers the already-grounded
answer in small chunks: ``sources`` first, then ``token`` events, then
``done`` — enough for progressive rendering without a streaming backend.
"""
from __future__ import annotations

import json
import time
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas import AskRequest, AskResponse, InjectionMeta, RAGMeta, SourceModel

router = APIRouter(prefix="/rag", tags=["rag"])

TOKENS_PER_EVENT = 3  # words per SSE "token" event


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_answer(answer: str) -> Iterator[str]:
    """Yield the answer in small multi-word chunks (whitespace-normalised)."""
    words = answer.split()
    for i in range(0, len(words), TOKENS_PER_EVENT):
        yield " ".join(words[i:i + TOKENS_PER_EVENT])


@router.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(payload: AskRequest, request: Request):
    registry = request.app.state.registry
    started = time.perf_counter()
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

    if payload.stream:
        def event_stream() -> Iterator[str]:
            yield _sse("sources", {
                "sources": result["sources"],
                "retrieval_mode": result["retrieval_mode"],
            })
            for chunk in _chunk_answer(result["answer"]):
                yield _sse("token", {"text": chunk})
            yield _sse("done", {
                "took_ms": took_ms,
                "request_id": request_id,
                "cache": meta_source.get("cache", "off"),
            })

        return StreamingResponse(
            event_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
