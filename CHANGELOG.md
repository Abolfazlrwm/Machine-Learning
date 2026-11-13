# Changelog

All notable changes to this portfolio are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versioning follows SemVer on the repository-level tags (`v*`).

## [2.5.0] — 2026-11-13

### Added — rag-farsi-assistant
- **Prompt-injection guardrails** (`src/guardrails.py`, OWASP LLM01):
  deterministic bilingual (fa/en) rule set — instruction override, prompt
  exfiltration, filter bypass, jailbreak lexicon, credential fishing — with
  weighted scoring (capped at 1.0) and a configurable threshold. Policies on
  `RAG_GUARDRAILS`: `flag` (default: answer anyway, attach
  `meta["injection"]` with score + matched rules) or `block` (refuse before
  retrieval and the LLM with a grounded Persian refusal). Weak lexicon
  matches need corroboration, so benign questions containing «نادیده» stay
  below the threshold — pinned by false-positive tests.
- **LexicalReranker** (`src/rerank.py`): offline, dependency-free pointwise
  re-ranker — rarity-weighted query-term coverage over the candidate batch
  (`w(t) = log(1 + N/df(t))`). Same `BaseReranker` call site as the neural
  cross-encoder, so swapping is a config change. Available as
  `RAG_RERANKER=lexical`.
- **nDCG@k metric** (`evaluation/metrics.py`): binary-relevance nDCG
  alongside Hit@k / MRR in the retrieval harness.

### Changed — evaluation
- `evaluate.py --compare` adds a `hybrid+rerank` column (hybrid candidates
  re-ordered by the lexical re-ranker on the shared index; `--no-rerank` to
  skip). On the CI-reproducible baseline it lifts hybrid hit@1 0.80 → 0.90
  and MRR 0.894 → 0.944, matching BM25 while fusing both signals — see the
  new committed report `evaluation/results/eval_report_ci.{md,json}`.
- Test suites grew 132 → 162 (rag) and 34 → 36 (serving): guardrail rules +
  pipeline policies, reranker scoring, nDCG fixtures, API meta contract.

### Added — repository
- **MkDocs documentation site** (`mkdocs.yml` + `docs/`, Material theme,
  strict mode): RAG architecture / retrieval / evaluation / guardrails,
  serving endpoints & observability, deployment shapes, stock methodology —
  all numbers pulled from the committed reports. Deployed to GitHub Pages on
  every push touching `docs/` via the new `Docs` workflow
  (build `--strict` → `actions/deploy-pages`, concurrency-guarded).

## [2.4.0] — 2026-11-11

### Added — rag-farsi-assistant
- **Semantic answer cache** (`src/semantic_cache.py`): the exact-match TTL
  cache upgraded with embedding-similarity lookup — a paraphrase whose folded
  question is cosine-similar >= threshold (default 0.92) to a cached one is
  served with a single query embedding instead of the full retrieval +
  generation path. Namespaces fold retrieval settings into the cache identity,
  byte-equal vectors short-circuit to similarity 1.0 (immune to float
  rounding), and near-miss similarities are reported so threshold tuning is
  measurable. Off by default; enable with `RAG_SEMANTIC_CACHE=1`.
- `meta` blocks carry `cache_similarity` on semantic hits; `pipeline.stats()`
  exposes the cache counters (kind, threshold, hits/misses/hit_rate).

### Added — serving/
- **RAG business metrics** on `/v1/metrics`:
  `rag_cache_requests_total{result=hit|miss|off}` and
  `rag_stage_duration_seconds` histograms (`stage=retrieve|generate`),
  recorded from the pipeline meta block into the injectable registry
  (`app.state.metrics`), not the module singleton.

### Added — observability & tooling
- **Prometheus + Grafana compose stack** (`--profile observability`): scrape
  config for the API's zero-dependency metrics endpoint, Grafana
  provisioning-as-code (datasource + dashboard) and the provisioned
  `RAG API Overview` dashboard — cache hit rate, p50/p95, per-route traffic,
  5xx rate, per-stage latency and cache outcomes, all plain PromQL.
- **Offline latency benchmark** (`serving/bench.py`): drives the real app
  over an ASGI transport (fake embedder, extractive answers, no network)
  across health / cache-off / cache-exact / cache-semantic; the committed
  first report (`serving/benchmarks/report.md`, n=200 c=10) shows the exact
  cache capped at the repeat share (84% hit rate) while the semantic cache
  absorbs the paraphrases too (96%).

### Changed
- CI: the RAG suite now runs on Python 3.12 **and** 3.13 (fail-fast off) with
  pytest-cov — term-missing in the job log, `coverage.xml` artifact from the
  3.12 run. New `security` job: bandit (medium+ severity gates) over the two
  shipped packages and pip-audit against a fresh venv holding exactly the
  serving runtime; the whole workflow also runs weekly (Mondays 05:00 UTC).
- Fixed a broken `branches:` push filter in the workflow.
- `hashlib` calls use `usedforsecurity=False` (the MD5 hashing-trick tokens
  and SHA1 cache keys are not security primitives) — bandit-clean.
- serving version bumped to 2.4.0; compose image tags aligned.

### Meta
- 194 tests across the three suites (RAG 132, stock 28, serving 34) — up
  from 171. READMEs, `.env.example` and roadmap updated.

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
- Metric route labels are version-stable across FastAPI releases
  (0.12x reports mounted paths on `scope["route"]`, 0.16x reports
  router-relative ones) — the middleware re-attaches the URL prefix.

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
