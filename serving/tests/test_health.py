"""Meta endpoints: root info and aggregate health."""
from __future__ import annotations

from serving.app import __version__


def test_root_lists_api_surface(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "ML Portfolio API"
    assert body["version"] == __version__
    assert set(body["endpoints"]) == {
        "/v1/rag/ask", "/v1/sms/classify", "/v1/stock/signal", "/v1/metrics"
    }
    assert body["health"] == "/v1/health"


def test_health_reports_all_services(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["services"] == {
        "rag": "available", "sms": "available", "stock": "available"
    }
    assert body["uptime_seconds"] >= 0


def test_process_time_header_present(client):
    resp = client.get("/v1/health")
    assert "x-process-time-ms" in {k.lower() for k in resp.headers}
