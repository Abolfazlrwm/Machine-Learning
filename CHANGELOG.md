# Changelog

All notable changes to this portfolio are documented here.
The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and versioning follows SemVer on the repository-level tags (`v*`).

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
