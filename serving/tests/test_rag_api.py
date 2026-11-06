"""RAG /ask endpoint: JSON mode, validation, and SSE streaming mode."""
from __future__ import annotations

import json


def test_ask_returns_grounded_answer_with_sources(client):
    resp = client.post("/v1/rag/ask", json={"question": "بیش‌برازش چیست؟"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "بیش‌برازش چیست؟"
    assert "[1]" in body["answer"] and "[2]" in body["answer"]  # citations kept
    assert len(body["sources"]) == 2
    src = body["sources"][0]
    assert {"title", "chunk_id", "score", "snippet"} == set(src)
    assert 0.0 <= src["score"] <= 1.0
    assert body["retrieval_mode"] == "hybrid"
    assert isinstance(body["took_ms"], int)


def test_ask_validates_payload(client):
    assert client.post("/v1/rag/ask", json={"question": ""}).status_code == 422
    assert client.post("/v1/rag/ask", json={}).status_code == 422
    too_long = {"question": "x" * 2001}
    assert client.post("/v1/rag/ask", json=too_long).status_code == 422
    bad_top_k = {"question": "ok؟", "top_k": 0}
    assert client.post("/v1/rag/ask", json=bad_top_k).status_code == 422


def test_ask_streams_server_sent_events(client):
    resp = client.post(
        "/v1/rag/ask", json={"question": "یادگیری عمیق چیست؟", "stream": True}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = {}
    for block in resp.text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(l for l in lines if l.startswith("event: ")).removeprefix("event: ")
        data = next(l for l in lines if l.startswith("data: ")).removeprefix("data: ")
        events.setdefault(event, []).append(json.loads(data))

    # sources first, tokens in the middle, done last
    assert "sources" in events and "token" in events and "done" in events
    sources_event = events["sources"][0]
    assert sources_event["retrieval_mode"] == "hybrid"
    assert len(sources_event["sources"]) == 2

    rejoined = " ".join(t["text"] for t in events["token"])
    expected = " ".join(
        "بیش‌برازش زمانی رخ می‌دهد که مدل داده آموزش را حفظ کند [1] "
        "و روی داده‌های ندیده ضعیف عمل کند [2].".split()
    )
    assert rejoined == expected
    assert isinstance(events["done"][0]["took_ms"], int)
