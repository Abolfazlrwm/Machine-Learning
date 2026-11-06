"""Stock signal endpoint (persisted walk-forward models)."""
from __future__ import annotations


def test_signal_returns_next_day_prediction(client):
    resp = client.get("/v1/stock/signal", params={"ticker": "AAPL"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["signal"] in ("up", "down")
    assert 0.0 <= body["probability_up"] <= 1.0
    assert body["model"] == "Random_Forest"
    # as_of must parse as an ISO timestamp
    from datetime import datetime

    datetime.fromisoformat(body["as_of"])
    assert isinstance(body["took_ms"], int)


def test_signal_lowercases_and_normalises_ticker(client):
    resp = client.get("/v1/stock/signal", params={"ticker": "aapl"})
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


def test_signal_unknown_ticker_is_404_with_error_code(client):
    resp = client.get("/v1/stock/signal", params={"ticker": "ZROZ"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_signal_requires_ticker(client):
    assert client.get("/v1/stock/signal").status_code == 422
