"""The real ServiceRegistry against an artifact-less root.

Guarantees the production failure mode: clean 503/404 with machine-readable
codes and fix-it hints — never a stack trace. Runs without heavy deps.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from serving.app.main import create_app
from serving.app.services import ServiceRegistry


def test_empty_repo_root_reports_degraded_health(tmp_path: Path):
    app = create_app(registry=ServiceRegistry(root=tmp_path))
    client = TestClient(app)
    body = client.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert set(body["services"].values()) == {"unavailable"}


def test_rag_endpoint_503_with_hint_when_no_artifacts(tmp_path: Path):
    app = create_app(registry=ServiceRegistry(root=tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/rag/ask", json={"question": "هر چیزی"})
    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "SERVICE_UNAVAILABLE"
    assert "hint" in error


def test_sms_endpoint_503_when_artifact_missing(tmp_path: Path):
    app = create_app(registry=ServiceRegistry(root=tmp_path))
    client = TestClient(app)
    resp = client.post("/v1/sms/classify", json={"text": "hello"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_stock_endpoint_404_when_no_cached_data(tmp_path: Path):
    app = create_app(registry=ServiceRegistry(root=tmp_path))
    client = TestClient(app)
    resp = client.get("/v1/stock/signal", params={"ticker": "AAPL"})
    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert "main.py" in error["hint"]


def test_real_repo_root_probes_have_all_keys():
    """Probe the actual repo: statuses() must always report three keys."""
    statuses = ServiceRegistry().statuses()
    assert set(statuses) == {"rag", "sms", "stock"}
    assert all(v in ("available", "unavailable") for v in statuses.values())
