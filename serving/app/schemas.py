"""Pydantic request/response models for the /v1 API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- RAG
class AskRequest(BaseModel):
    """One grounded question for the Persian RAG assistant."""

    question: str = Field(
        min_length=1, max_length=2000,
        examples=["بیش‌برازش (overfitting) چیست و چطور جلوی آن را گرفت؟"],
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Override the number of retrieved chunks (1-20).",
    )
    stream: bool = Field(
        default=False,
        description="When true, answer arrives as server-sent events (SSE).",
    )


class SourceModel(BaseModel):
    title: str
    chunk_id: str
    score: float
    snippet: str


class InjectionMeta(BaseModel):
    """Prompt-injection screening summary (src/guardrails.py in the RAG project)."""

    flagged: bool
    score: float = Field(ge=0.0, le=1.0)
    rules: list[str]


class RAGMeta(BaseModel):
    """Observability block for one /v1/rag/ask call."""

    request_id: str
    cache: Literal["hit", "miss", "off"] = "off"
    timings_ms: dict[str, int] | None = None
    query_expansions: list[str] | None = None
    mixed_script: bool | None = None
    guardrails: Literal["flag", "block"] | None = None
    injection: InjectionMeta | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    retrieval_mode: str
    took_ms: int
    meta: RAGMeta | None = None


# --------------------------------------------------------------------- SMS
class SmsRequest(BaseModel):
    text: str = Field(
        min_length=1, max_length=2000,
        examples=["Free entry in 2 a wkly comp to win FA Cup final tkts!"],
    )


class SmsResponse(BaseModel):
    label: Literal["spam", "ham"]
    probability_spam: float = Field(ge=0.0, le=1.0)
    model_version: str
    took_ms: int


# ------------------------------------------------------------------- stock
class StockSignalResponse(BaseModel):
    ticker: str
    signal: Literal["up", "down"]
    probability_up: float = Field(ge=0.0, le=1.0)
    model: str
    as_of: datetime = Field(description="Date of the last feature row used.")
    took_ms: int


# ------------------------------------------------------------------ health
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    services: dict[str, Literal["available", "unavailable"]]
    uptime_seconds: float
