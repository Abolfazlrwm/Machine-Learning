# Observability

The serving layer ships with a business-metrics layer, not just default
HTTP counters: the things you actually tune in a RAG system are **cache
effectiveness** and **where the latency lives**.

## Metrics exposed

| metric | type | labels | answers |
|---|---|---|---|
| `rag_cache_requests` | counter | `result` (`hit`/`miss`/`off`) | is the semantic cache earning its keep? |
| `rag_stage_duration_seconds` | histogram | `stage` (`retrieve`/`generate`/…) | where does p95 latency come from? |
| standard HTTP counters/histograms | — | method, path, status | traffic and error rates |

Each `/v1/rag/ask` call increments the cache counter from the pipeline's
`meta.cache` outcome and observes the stage timings (milliseconds in the
meta block, seconds in the histograms — the registry converts).

## Grafana dashboard

The Docker Compose stack provisions Prometheus + Grafana automatically
(`docker-compose.yml` + provisioned datasource and dashboard JSON): the
dashboard opens pre-wired with cache hit-rate and stage-latency panels —
no manual dashboard building after `docker compose up`.

```bash
cd serving
docker compose up --build
# API      http://localhost:8000
# metrics  http://localhost:8000/metrics
# Grafana  http://localhost:3000  (anonymous viewer)
```

## Semantic cache

`src/semantic_cache.py` upgrades the exact-match TTL cache: the question
is embedded once and a cached answer is reused when cosine similarity
clears the threshold (default 0.92). One query embedding replaces the
entire retrieval + generation path, so the break-even is met quickly; the
cache namespace keys on `top_k`, retrieval mode, re-ranker, embedder and
index size, so a stale answer can never leak across configurations.

## Request tracing

Every pipeline stage runs inside a `Trace` span (`src/observability.py`);
the timings land in `meta.timings_ms`, which is what the histograms
consume. The same spans drive the evaluation harness, so a latency
regression shows up in the committed benchmark report too
(`serving/benchmarks/report.md`).
