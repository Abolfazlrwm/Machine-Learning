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

### SSE streaming — real-time tokens

`{"stream": true}` switches the response to `text/event-stream`. Since
v2.6.0 the stream is *real*: OpenAI-compatible and Ollama providers yield
LLM deltas as they are generated, and the router maps the pipeline's
events 1:1 onto SSE frames. Backends without a streaming protocol still
work — the pipeline re-chunks the ready answer into small `token` events —
so the contract is identical either way.

| event | payload |
|---|---|
| `sources` | `{sources, retrieval_mode, guardrails?, injection?}` — grounding first, so citations render before the first token |
| `token` | `{text}` — answer fragments; concatenating them reproduces the full answer |
| `done` | `{took_ms, request_id, cache, meta}` — stage timings, query plan, cache outcome |
| `error` | `{code, message, hint?}` — mid-stream failures, same contract as the JSON error model |

```bash
curl -N -X POST localhost:8000/v1/rag/ask \
  -H 'content-type: application/json' \
  -d '{"question": "بیش‌برازش چیست؟", "stream": true}'
```

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
