# Serving API

`serving/` wraps all three projects behind one FastAPI application with a
versioned REST contract, optional SSE streaming, API-key auth and
Prometheus metrics. The RAG service imports the assistant's modules
directly (no subprocess hop) and degrades cleanly: if a heavy backend is
missing, the endpoint answers **503** with a hint, not a stack trace.

## Endpoints

| method | path | purpose |
|---|---|---|
| `GET` | `/v1/health` | service registry status per project |
| `POST` | `/v1/rag/ask` | grounded Persian QA — JSON body `{question, top_k?, stream?}` |
| `POST` | `/v1/sms/classify` | spam/ham classification |
| `GET` | `/v1/stock/signal` | next-day movement signal for a persisted ticker |

### `/v1/rag/ask` — JSON mode

```bash
curl -s -X POST localhost:8000/v1/rag/ask \
  -H 'content-type: application/json' \
  -d '{"question": "بیش‌برازش چیست؟"}'
```

```json
{
  "question": "بیش‌برازش چیست؟",
  "answer": "بیش‌برازش زمانی رخ می‌دهد که مدل داده آموزش را حفظ کند [1] …",
  "sources": [{"title": "…", "chunk_id": "…", "score": 0.91, "snippet": "…"}],
  "retrieval_mode": "hybrid",
  "took_ms": 34,
  "meta": {
    "request_id": "…",
    "cache": "miss",
    "timings_ms": {"retrieve": 9, "generate": 11},
    "query_expansions": ["overfitting"],
    "mixed_script": false,
    "injection": null
  }
}
```

The `meta` block is the observability contract: cache outcome, per-stage
latency, query-expansion transparency and — when the guardrails flag
something — the [injection report](../rag/safety.md).

### SSE streaming

`{"stream": true}` switches the response to `text/event-stream`: a
`sources` event first, then small `token` events, then `done` with
`took_ms`. The backing LLM providers are non-streaming, so this delivers
progressive rendering of the already-grounded answer — enough for chat UX
without a streaming backend.

## Configuration

| env | default | meaning |
|---|---|---|
| `RAG_EMBEDDING_BACKEND` | `sentence-transformers` | set `fake` for a no-download smoke deployment |
| `RAG_GUARDRAILS` | `flag` | injection policy surfaced through `meta` |
| `API_KEY` / auth mode | none | API-key gate on `/v1/*` |

## Run it

```bash
cd serving
pip install -r requirements.txt
uvicorn app.main:app --reload           # local

# or the full stack (API + Prometheus + Grafana):
docker compose -f docker-compose.yml up --build
```
