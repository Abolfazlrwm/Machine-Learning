# Serving API — offline latency benchmark

Single-process ASGI benchmark (`httpx.ASGITransport`) of the real FastAPI
app: RAG on the deterministic fake embedder, extractive answers, no
network and no model inference. Measures middleware + query intelligence
+ BM25/vector retrieval + RRF + caching + serialisation.

- workload: `200` requests at concurrency `10` — 80% repeats of an
  8-question Persian pool, 20% paraphrases (prefix + word-order swap)
- machine: benchmark runner inside CI-equivalent container, one process
- hit rate: parsed from the responses' `meta.cache`, never assumed

| scenario | description | cache hit | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | max ms | req/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `health` | GET /v1/health — stack floor, no model work | 0% | 4.7 | 5.4 | 5.6 | 6.1 | 4.7 | 6.8 | 1558 |
| `cache-off` | POST /v1/rag/ask — cache disabled, full pipeline each time | 0% | 14.8 | 19.9 | 26.0 | 45.6 | 16.5 | 47.9 | 567 |
| `cache-exact` | POST /v1/rag/ask — exact TTL cache (repeats hit) | 84% | 12.4 | 14.9 | 15.8 | 17.2 | 12.3 | 19.3 | 749 |
| `cache-semantic` | POST /v1/rag/ask — semantic cache (repeats + paraphrases hit) | 96% | 15.5 | 19.5 | 22.1 | 26.0 | 15.9 | 28.1 | 589 |

## What the numbers say

- `cache-off` is the honest cost of one full hybrid retrieval per request;
- `cache-exact` cuts latency for repeated questions but every paraphrase
  still pays the full price — hit rate stays at the repeat share;
- `cache-semantic` absorbs the paraphrases too (the hit rate climbs to the
  repeat + paraphrase share) while adding one query embedding per miss.

Single process and GIL-bound: absolute req/s understates a deployed
multi-worker setup — read the columns *relative* to `cache-off`.
