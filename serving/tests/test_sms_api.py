"""SMS spam classification endpoint."""
from __future__ import annotations


def test_classifies_spam(client):
    resp = client.post(
        "/v1/sms/classify",
        json={"text": "WIN a free prize now! claim your reward"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "spam"
    assert body["probability_spam"] > 0.5
    assert body["model_version"]
    assert isinstance(body["took_ms"], int)


def test_classifies_ham(client):
    resp = client.post("/v1/sms/classify", json={"text": "Hey, are we still on for lunch tomorrow?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "ham"
    assert body["probability_spam"] < 0.5


def test_classify_validates_payload(client):
    assert client.post("/v1/sms/classify", json={"text": ""}).status_code == 422
    assert client.post("/v1/sms/classify", json={}).status_code == 422
    assert client.post("/v1/sms/classify", json={"text": "x" * 2001}).status_code == 422
