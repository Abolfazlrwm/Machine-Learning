# Serving API — one FastAPI layer over the ML portfolio

A single versioned HTTP surface (`/v1`) for three model families of this
repository, designed the way production ML services are:

- **Lazy singletons with clean degradation** — every service probes its own
  artifacts (model, index, dataset) on the filesystem. Missing artifacts
  answer `503` with a machine-readable code + fix-it hint; the API boots
  anywhere, including CI.
- **Injectable registry** — `create_app(registry=...)` accepts fakes, which
  is why the whole test-suite runs network-free in ~2s.
- **SSE streaming** for the RAG endpoint, **optional API-key guard**, CORS,
  per-request `X-Process-Time-Ms`, and a stable error envelope:
  `{"error": {"code": ..., "message": ..., "hint": ...}}`.

```
serving/
├── app/
│   ├── main.py        # app factory: routing, auth, CORS, health, errors
│   ├── schemas.py     # pydantic request/response models
│   ├── services.py    # lazy RAG / SMS / stock services + registry
│   ├── exceptions.py  # APIError -> 502/503/404 with stable codes
│   └── routers/       # rag.py (ask + SSE), sms.py, stock.py
└── tests/             # 21 network-free tests (fakes + unavailable paths)
```

## Endpoints

| Endpoint | Method | Body / Query | Response |
|---|---|---|---|
| `/v1/rag/ask` | POST | `{"question": "...", "top_k": 4, "stream": false}` | `AskResponse` or SSE stream |
| `/v1/sms/classify` | POST | `{"text": "..."}` | `{"label": "spam"/"ham", "probability_spam": ...}` |
| `/v1/stock/signal` | GET | `?ticker=AAPL` | `{"signal": "up"/"down", "probability_up": ..., "as_of": ...}` |
| `/v1/health` | GET | — | `{"status": "ok"/"degraded", "services": {...}}` |
| `/v1/metrics` | GET | — | Prometheus text exposition (v0.0.4) — request counters + latency histograms |
| `/` | GET | — | API index (version, endpoint list) |

Every response carries an `X-Request-ID` header (the caller's value is
echoed, otherwise a `uuid4` hex is generated) and `X-Process-Time-Ms`.
The RAG `ask` response additionally embeds a `meta` block: request id,
answer-cache state (`hit` / `miss` / `off`), per-stage retrieval timings
in milliseconds and the query-expansion terms that were appended.

### RAG ask — JSON

```bash
curl -s localhost:8000/v1/rag/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "بیش‌برازش چیست و چطور جلوی آن را گرفت؟", "top_k": 4}'
```

```json
{
  "question": "بیش‌برازش چیست و چطور جلوی آن را گرفت؟",
  "answer": "…[1] …[2]",
  "sources": [
    {"title": "01_machine_learning_basics.txt",
     "chunk_id": "01_machine_learning_basics:03",
     "score": 0.9123, "snippet": "…"}
  ],
  "retrieval_mode": "hybrid",
  "took_ms": 148
}
```

### RAG ask — SSE (`"stream": true`)

The backing LLM providers are non-streaming, so the endpoint grounds the
answer first and then emits three event types:

```
event: sources   → {"sources": [...], "retrieval_mode": "hybrid"}
event: token     → {"text": "چند کلمه از پاسخ"}   (repeated)
event: done      → {"took_ms": 148}
```

```bash
curl -sN localhost:8000/v1/rag/ask -H 'Content-Type: application/json' \
     -d '{"question": "بیش‌برازش چیست؟", "stream": true}'
```

### Stock signal

The stock pipeline persists its trained models (`models/{TICKER}_{name}.joblib`,
written by `src.train.persist_models`). The service recomputes the 20 technical
indicators on the latest cached OHLCV row and returns the next-day signal:

```bash
curl -s "localhost:8000/v1/stock/signal?ticker=AAPL"
# {"ticker":"AAPL","signal":"up","probability_up":0.5734,"model":"Random_Forest",
#  "as_of":"2026-11-05T00:00:00","took_ms":36}
```

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `ML_REPO_ROOT` | auto | Repo root (auto-detected two levels up) |
| `SERVING_API_KEY` | unset | If set, `/v1/*` requires `X-API-Key` (health stays open) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `RAG_*` | see RAG `.env.example` | All RAG knobs (backend, provider, retrieval mode, …) |

## Docker

The root `Dockerfile` is multi-stage (builder venv → slim runtime, non-root,
built-in healthcheck) and installs the full stack
(`serving/requirements-docker.txt`):

```bash
docker compose up --build             # API on :8000 (RAG + sms + stock)
docker compose --profile llm up       # + Ollama (pull llama3.1 once)
docker compose --profile ui up        # + Streamlit chat on :8501
```

Volumes: the RAG index persists in a named volume; stock data/models and the
SMS artifact are bind-mounted read-only — build them on the host first:

```bash
python stock-movement-ml/main.py --ticker AAPL     # data + models/ + reports/
cd sms-spam && python spam_classifier.py --train spam.csv   # spam_classifier.pkl
```

## Tests

```bash
cd serving && pip install -r requirements-dev.txt
python -m pytest tests/ -q      # 21 tests: routes, SSE, auth, 422/404/503 paths
```
