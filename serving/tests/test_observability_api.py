"""Observability endpoints: /v1/metrics, X-Request-ID and the RAG meta block.

Uses self-contained fakes (not the shared conftest ones) because the plain
``tests`` package name is already taken by the RAG project's suite once
``serving.app.metrics`` has put ``rag-farsi-assistant`` on ``sys.path``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serving.app.main import create_app  # noqa: E402
from serving.app.metrics import MetricsRegistry  # noqa: E402  (re-export)
from serving.app.services import ServiceStatus  # noqa: E402


class MetaFakeRAG:
    """Fake RAG service that mimics the pipeline's meta block."""

    name = "rag"

    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> ServiceStatus:
        return ServiceStatus(True, "fake")

    def ask(self, question: str, top_k: int | None = None) -> dict:
        self.calls += 1
        return {
            "question": question,
            "answer": "پاسخ آزمایشی با ارجاع [1].",
            "sources": [{
                "title": "01_machine_learning_basics.txt",
                "chunk_id": "01_machine_learning_basics:03",
                "score": 0.9123,
                "snippet": "بیش‌برازش…",
            }],
            "retrieval_mode": "hybrid",
            "meta": {
                "cache": "miss" if self.calls == 1 else "hit",
                "timings_ms": {"bm25": 3, "dense": 5, "rrf": 0, "generate": 11},
                "query": {
                    "original": question,
                    "expansions": ["overfitting"],
                    "mixed_script": True,
                },
            },
        }


class _StaticRegistry:
    """Registry stand-in: only what create_app and the rag router touch."""

    def __init__(self) -> None:
        self.rag = MetaFakeRAG()

    def statuses(self) -> dict[str, str]:
        return {"rag": "available", "sms": "available", "stock": "available"}


@pytest.fixture()
def fresh_metrics() -> MetricsRegistry:
    """An isolated registry so assertions see only their own traffic."""
    return MetricsRegistry()


@pytest.fixture()
def api(fresh_metrics) -> TestClient:
    # Fresh registry per test so MetaFakeRAG's call counter (cache miss/hit
    # simulation) starts clean in every test.
    return TestClient(create_app(registry=_StaticRegistry(), metrics=fresh_metrics))


# ------------------------------------------------------------- /v1/metrics
def test_metrics_endpoint_serves_prometheus_text(api):
    resp = api.get("/v1/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in resp.headers["content-type"]


def test_requests_are_counted_with_route_templates(api, fresh_metrics):
    api.get("/v1/health")
    api.post("/v1/rag/ask", json={"question": "بیش‌برازش چیست؟"})
    text = fresh_metrics.render()
    assert 'http_requests_total{method="GET",path="/v1/health",status="200"} 1' in text
    assert 'http_requests_total{method="POST",path="/v1/rag/ask",status="200"} 1' in text


def test_latency_histogram_has_sum_and_count(api, fresh_metrics):
    api.get("/v1/health")
    text = fresh_metrics.render()
    assert "# TYPE http_request_duration_seconds histogram" in text
    assert re.search(r'http_request_duration_seconds_bucket\{[^}]*le="\+Inf"[^}]*\} 1', text)
    assert 'http_request_duration_seconds_count{method="GET",path="/v1/health"} 1' in text
    assert "http_request_duration_seconds_sum" in text


def test_errors_are_counted_by_status(api, fresh_metrics):
    api.post("/v1/rag/ask", json={"question": ""})  # 422
    text = fresh_metrics.render()
    assert 'http_requests_total{method="POST",path="/v1/rag/ask",status="422"} 1' in text


# ------------------------------------------------------------ X-Request-ID
def test_request_id_is_echoed_when_provided(api):
    resp = api.get(
        "/v1/health", headers={"X-Request-ID": "my-trace-id-1234"}
    )
    assert resp.headers["X-Request-ID"] == "my-trace-id-1234"


def test_request_id_is_generated_when_absent(api):
    resp = api.get("/v1/health")
    value = resp.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", value)  # uuid4().hex


# -------------------------------------------------------------- RAG meta
def test_ask_response_carries_meta_block(api):
    body = api.post("/v1/rag/ask", json={"question": "بیش‌برازش چیست؟"}).json()
    meta = body["meta"]
    assert meta["cache"] == "miss"
    assert meta["timings_ms"]["generate"] == 11
    assert meta["query_expansions"] == ["overfitting"]
    assert meta["mixed_script"] is True
    assert re.fullmatch(r"[0-9a-f]{32}", meta["request_id"])
    assert meta["request_id"] == api.headers.get("x-request-id", meta["request_id"])


def test_ask_meta_reports_cache_hits(api):
    api.post("/v1/rag/ask", json={"question": "بیش‌برازش چیست؟"})
    body = api.post("/v1/rag/ask", json={"question": "بیش‌برازش چیست؟"}).json()
    assert body["meta"]["cache"] == "hit"


def test_sse_done_event_carries_request_id_and_cache_state(api):
    resp = api.post(
        "/v1/rag/ask", json={"question": "بیش‌برازش چیست؟", "stream": True}
    )
    done = [b for b in resp.text.strip().split("\n\n") if "event: done" in b][0]
    assert '"request_id"' in done and '"cache": "miss"' in done
    assert resp.headers["X-Request-ID"]


def test_root_lists_metrics_endpoint(api):
    assert "/v1/metrics" in api.get("/").json()["endpoints"]
