"""Shared fixtures: fake services + app factory.

The API is designed around an injectable :class:`ServiceRegistry`, so tests
swap the real (heavy) services for deterministic fakes. A separate test
module exercises the *real* registry's unavailable path against an empty
directory — no heavy dependencies needed anywhere.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.app.exceptions import NotFound  # noqa: E402
from serving.app.main import create_app  # noqa: E402
from serving.app.services import ServiceStatus  # noqa: E402


# ------------------------------------------------------------------ fakes
class FakeRAG:
    name = "rag"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> ServiceStatus:
        return ServiceStatus(True, "fake")

    def ask(self, question: str, top_k: int | None = None) -> dict:
        self.calls += 1
        meta: dict = {
            "cache": "miss" if self.calls == 1 else "hit",
            "timings_ms": {"bm25": 3, "dense": 5, "rrf": 0, "generate": 11},
            "query": {
                "original": question,
                "expansions": ["overfitting"],
                "mixed_script": True,
            },
        }
        # Mirror the real pipeline contract (guardrails="flag"): flagged
        # questions keep their answer but carry the injection report.
        injection_trigger = "نادیده بگیر" in question or "system prompt" in question.lower()
        if injection_trigger:
            meta["guardrails"] = "flag"
            meta["injection"] = {
                "flagged": True,
                "score": 1.0,
                "rules": ["override-fa" if "نادیده بگیر" in question else "probe-system-prompt-en"],
            }
        return {
            "question": question,
            "answer": (
                "بیش‌برازش زمانی رخ می‌دهد که مدل داده آموزش را حفظ کند [1] "
                "و روی داده‌های ندیده ضعیف عمل کند [2]."
            ),
            "sources": [
                {
                    "title": "01_machine_learning_basics.txt",
                    "chunk_id": "01_machine_learning_basics:03",
                    "score": 0.9123,
                    "snippet": "بیش‌برازش زمانی رخ می‌دهد که مدل…",
                },
                {
                    "title": "02_deep_learning_neural_networks.txt",
                    "chunk_id": "02_deep_learning_neural_networks:01",
                    "score": 0.7412,
                    "snippet": "regularization and validation…",
                },
            ],
            "retrieval_mode": "hybrid",
            "meta": meta,
        }

    def ask_stream(self, question: str, top_k: int | None = None):
        """Mirror the real ``RAGService.ask_stream`` event contract."""
        result = self.ask(question, top_k=top_k)
        meta = result["meta"]
        sources_event: dict = {
            "type": "sources",
            "sources": result["sources"],
            "retrieval_mode": result["retrieval_mode"],
        }
        if meta.get("injection"):
            sources_event["guardrails"] = meta["guardrails"]
            sources_event["injection"] = meta["injection"]
        yield sources_event
        words = result["answer"].split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i + 3])
            if i + 3 < len(words):
                chunk += " "  # trailing space keeps "".join reconstruction exact
            yield {"type": "token", "text": chunk}
        yield {
            "type": "done",
            "cache": meta.get("cache", "off"),
            "took_ms": 42,
            "meta": {
                "timings_ms": meta.get("timings_ms"),
                "query": meta.get("query"),
            },
        }


class FakeSMS:
    name = "sms"

    def available(self) -> ServiceStatus:
        return ServiceStatus(True, "fake")

    def classify(self, text: str) -> dict:
        spammy = any(w in text.lower() for w in ("win", "free", "prize", "claim"))
        return {
            "label": "spam" if spammy else "ham",
            "probability_spam": 0.9312 if spammy else 0.0210,
            "model_version": "fake-tfidf-logreg",
        }


class FakeStock:
    name = "stock"

    def available(self) -> ServiceStatus:
        return ServiceStatus(True, "fake")

    def signal(self, ticker: str) -> dict:
        if ticker.upper() == "ZROZ":  # exercised 404 path
            raise NotFound("no persisted model for ticker 'ZROZ'")
        return {
            "ticker": ticker.upper(),
            "signal": "up",
            "probability_up": 0.5734,
            "model": "Random_Forest",
            "as_of": datetime(2026, 11, 5),
        }


@dataclass
class FakeRegistry:
    rag: object = field(default_factory=FakeRAG)
    sms: object = field(default_factory=FakeSMS)
    stock: object = field(default_factory=FakeStock)

    def statuses(self) -> dict[str, str]:
        return {"rag": "available", "sms": "available", "stock": "available"}


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def fake_registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture(scope="session")
def client(fake_registry) -> TestClient:
    return TestClient(create_app(registry=fake_registry))


@pytest.fixture(scope="session")
def secure_client(fake_registry) -> TestClient:
    return TestClient(create_app(registry=fake_registry, api_key="topsecret"))
