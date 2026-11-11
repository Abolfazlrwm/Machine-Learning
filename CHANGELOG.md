# Changelog

All notable changes to this portfolio are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versioning follows SemVer on the repository-level tags (`v*`).

## [2.3.0] — 2026-11-11

### Added — rag-farsi-assistant
- **Bilingual query understanding** (`src/query.py`): every question is
  folded into the exact canonical form the BM25 tokenizer produces (NFC,
  Arabic->Persian characters, ZWNJ stripped, casefolded), profiled for
  Persian/Latin script mix, and expanded with cross-language counterparts
  from a curated ML glossary («بیش‌برازش» <-> overfitting, رانش داده <-> data
  drift, ...). Expansion feeds the **lexical sub-query only** — the dense
  sub-search embeds the visitor's original wording via the new
  `HybridRetriever(embed_text=...)`, so the evaluated E5 behaviour and the
  published golden-set numbers are untouched.
- **Stage tracing** (`src/observability.py::Trace`): cumulative per-stage
  timings for dense / bm25 / rrf / rerank plus candidate counts, attached
  to every answer as a `meta` block.
- **Zero-dependency Prometheus metrics** (`MetricsRegistry`): counters and
  histograms rendering the text exposition format (v0.0.4) from scratch —
  sorted escaped labels, cumulative buckets, +Inf/sum/count series,
  thread-safe. No `prometheus_client`.
- **TTL + LRU answer cache** (`src/cache.py`): thread-safe, injectable
  clock, hit/miss/expiry/eviction stats. Cache identity = folded question +
  retrieval settings + index size; disabled by default (`cache_ttl = 0`)
  so evaluation stays deterministic.

### Added — serving/
- **`GET /v1/metrics`** exposing the registry in Prometheus text format.
- **`X-Request-ID` middleware**: echoes the caller's id or generates a
  `uuid4` hex, logged per request; `http_requests` counters and
  `http_request_duration_seconds` histograms per method / route template /
  status (bounded label cardinality).
- **RAG `meta` block** in the `ask` response (JSON) and the SSE `done`
  event: request id, cache hit/miss/off, per-stage timings in ms and the
  query-expansion terms that were appended.

### Fixed
- `RAGService._build` degrades a missing heavy backend (e.g. no
  sentence-transformers) to the usual `503` + fix-it hint instead of a
  stack trace.

### Meta
- 171 tests across three CI jobs (RAG 112, stock 28, serving 31) — up from
  121. Roadmap and READMEs updated; `serving` version bumped to 2.3.0.

## [2.2.0] — 2026-11-06

### Added — serving/ (new top-level project)
- **FastAPI serving layer** exposing all three model families behind one
  versioned API: `POST /v1/rag/ask` (JSON **and SSE streaming**),
  `POST /v1/sms/classify`, `GET /v1/stock/signal`, `GET /v1/health`.
- **Clean degradation**: services probe their own artifacts; missing
  models/indexes answer `503` with a machine-readable error code and a
  fix-it hint instead of a stack trace (unknown ticker → `404 NOT_FOUND`).
- **Optional API-key guard** (`SERVING_API_KEY` → `X-API-Key` header; health
  stays open for probes), CORS, per-request `X-Process-Time-Ms`.
- Injectable `ServiceRegistry` — the 21-test suite runs network-free in ~2s
  by swapping fakes in, and additionally exercises the real registry's
  unavailable paths against an empty root.

### Added — deployment
- **Multi-stage Dockerfile** (builder venv → slim non-root runtime with
  healthcheck) plus a `docker-compose.yml` stack: `api` + optional `ollama`
  and `streamlit` profiles; Streamlit image via a new
  `rag-farsi-assistant/Dockerfile`.
- **Hugging Face Spaces demo** (`hf-space/`): standalone Gradio app with E5
  dense retrieval, cited extractive answers and an automatic lexical
  fallback — ready to push to a Space for a live CV link.

### Changed — stock-movement-ml
- Trained models are now persisted (`src/train.py::persist_models` →
  `models/{TICKER}_{name}.joblib`) so the API can serve signals without
  retraining; covered by a round-trip test (28 tests, up from 27).

### Meta
- CI gained a third job (`serving-api`) — 121 tests total across three jobs.
- README: new Serving API + HF Spaces sections, roadmap and releases updated.

## [2.1.0] — 2026-11-05

### Added — rag-farsi-assistant
- **Hybrid retrieval**: BM25 (Okapi) implemented from scratch with
  Persian-aware tokenisation (Arabic Yeh/Keh normalisation, ZWNJ handling,
  built-in stopword list) — `src/bm25.py`.
- **Reciprocal Rank Fusion** (`src/hybrid.py`) combining dense and sparse
  rankings, with three switchable retrieval modes: `dense` / `sparse` / `hybrid`
  (`RAG_RETRIEVAL_MODE`, `--mode`, Streamlit sidebar).
- **Optional cross-encoder re-ranking hook** (`src/rerank.py`, e.g.
  `BAAI/bge-reranker-v2-m3`) behind a lazy import.
- **RAGAS-style evaluation suite** (`evaluation/metrics.py`): context
  precision, context recall, faithfulness, answer relevancy — deterministic,
  judge-free, fully reproducible.
- Golden QA set expanded from 10 to **30 pairs** (5 per corpus document) with
  gold answers; evaluation reports stored under `evaluation/results/`.
- 12 new test modules-cases covering BM25 ranking, Persian normalisation, RRF
  hand-computed scores, mode switching, reranker plumbing and the eval
  metrics (72 tests total, up from 33).

### Measured (`intfloat/multilingual-e5-base`, 30-pair golden set)
- hybrid (RRF): Hit@1 **0.967**, MRR **0.983** — best overall;
- dense (E5 + MMR): Hit@1 0.933, Hit@2/3 1.000, MRR 0.967;
- sparse (BM25): Hit@1 0.933, Hit@3 1.000, MRR 0.961 — dense and sparse each
  miss a different question and RRF recovers both;
- generation suite: doc hit rate 1.000, context precision 0.942,
  context recall 0.979, faithfulness 1.000, answer relevancy 0.846
  (extractive answer mode, scored with the E5 cosine).

## [2.0.0] — 2026-11-02

### Added
- Repository-level CI (GitHub Actions) running the RAG and stock test suites
  on every push/PR.
- Portfolio README indexing all five projects with real headline metrics.
- Root `.gitignore` for caches, secrets and artifacts.
- `sms-spam` renamed into its own folder with a proper script and README
  (Accuracy 0.975, ROC-AUC 0.987).

### Changed
- Master branches reorganised: five self-contained projects sharing one
  documented convention (`src/` + `tests/` + README per project).

## [1.0.0] — 2026-10-13

### Added
- Initial portfolio: `rag-farsi-assistant` v1 (E5 + FAISS + MMR + dual LLM
  provider, tagged `rag-v1.0.0`) and `stock-movement-ml` v1.1 (20 indicators,
  walk-forward backtest, tagged `stock-v1.1.0`).
