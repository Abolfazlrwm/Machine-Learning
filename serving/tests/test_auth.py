"""Optional API-key guard: /v1 is protected, health stays open for probes."""
from __future__ import annotations


def test_v1_rejects_missing_and_wrong_keys(secure_client):
    resp = secure_client.post(
        "/v1/sms/classify", json={"text": "hello there friend"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    resp = secure_client.post(
        "/v1/sms/classify",
        json={"text": "hello there friend"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_v1_accepts_valid_key(secure_client):
    resp = secure_client.post(
        "/v1/sms/classify",
        json={"text": "hello there friend"},
        headers={"X-API-Key": "topsecret"},
    )
    assert resp.status_code == 200
    assert resp.json()["label"] == "ham"


def test_health_and_docs_stay_open_without_key(secure_client):
    # k8s/uptime probes must not need credentials
    assert secure_client.get("/v1/health").status_code == 200
    assert secure_client.get("/").status_code == 200
