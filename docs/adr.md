# Engineering Decision Records

Why the load-bearing choices in this portfolio were made the way they were.
Each record follows the Nygard ADR format (Context → Decision → Consequences)
and refers to real, verified numbers from this repository — nothing here is
aspirational. For the resulting architecture, see the
[RAG architecture](rag/architecture.md) and the [deployment notes](deployment.md).

| # | Decision | One-line why |
|---|----------|--------------|
| [ADR-001](#adr-001-hybrid-retrieval-fused-with-rrf-bm25-from-scratch) | Hybrid retrieval with RRF, BM25 from scratch | dense misses lexemes, sparse misses paraphrases |
| [ADR-002](#adr-002-embedding-default-pinned-to-e5-small-with-a-contract-test) | Embedding default pinned to E5-small + contract test | a 768-vs-384 drift must be impossible to repeat |
| [ADR-003](#adr-003-network-free-ci-fake-embedder-mock-llm) | Network-free CI (fake embedder, mock LLM) | fast, deterministic, free — with contract tests |
| [ADR-004](#adr-004-a-bounded-re-rank-stage-after-fusion) | Bounded re-rank stage after fusion | +0.11 Hit@1 for a top-k cost, not a full-scan cost |
| [ADR-005](#adr-005-prompt-injection-guardrails-before-prompt-build) | Prompt-injection guardrails before prompt build | retrieved text is untrusted input |
| [ADR-006](#adr-006-two-layer-answer-cache-exact-ttl-semantic) | Two-layer answer cache (exact TTL + semantic) | repeated questions must not re-pay LLM cost |
| [ADR-007](#adr-007-sse-streaming-and-per-stage-prometheus-metrics) | SSE streaming + per-stage Prometheus metrics | users see tokens, operators see stages |
| [ADR-008](#adr-008-honest-numbers-or-no-numbers) | Honest numbers or no numbers | credibility is the portfolio's product |
| [ADR-009](#adr-009-raw-numpy-autograd-and-transformer-tinygpt) | Raw-NumPy autograd + transformer (TinyGPT) | prove the primitives, not just the APIs |
| [ADR-010](#adr-010-static-pyodidewasm-space-for-tinygpt) | Static Pyodide/WASM Space for TinyGPT | free tier runs the real stack in-browser |
| [ADR-011](#adr-011-one-fastapi-layer-over-all-model-families) | One FastAPI layer over all model families | one contract, one deployment path |

## ADR-001: Hybrid retrieval fused with RRF (BM25 from scratch)

**Status:** Accepted

**Context.** Dense retrieval (E5 + FAISS) generalises across paraphrases but
drops exact lexemes — identifiers, foreign terms, rare Persian words. Sparse
lexical retrieval catches those but misses meaning. Persian adds a twist:
tokenisation must respect ZWNJ, Arabic-script variants (ي/ی, ك/ک) and
mixed-script queries, and off-the-shelf engines make that a black box.

**Decision.** Fuse dense and sparse rankings with Reciprocal Rank Fusion.
BM25 is implemented from scratch with Persian-aware normalisation and
tokenisation rather than delegating to an external engine. On the committed
golden set, hybrid wins dense-only and sparse-only with
Hit@1 = 0.967 / MRR = 0.983 ([retrieval](rag/retrieval.md)).

**Consequences.** No JVM/Lucene dependency and full control of Persian text
handling; fusion is parameter-light (no learned weights to overfit); the cost
is maintaining our own scorer — acceptable because it is unit-tested against
known rankings.

## ADR-002: Embedding default pinned to E5-small, with a contract test

**Status:** Accepted

**Context.** A live API capture exposed a silent drift: the runtime had moved
to a 768-dim encoder while every published artifact — static Space embeddings,
storage metadata, Hit@1 evaluations — was built at 384 dims. Nothing crashed;
the pipeline quietly degraded.

**Decision.** The default is restored to `intfloat/multilingual-e5-small`
and pinned by a contract test that asserts the configured model's dimensionality
against the value the artifacts were built with.

**Consequences.** Switching encoders is now a conscious act (config change +
artifact rebuild + green contract test), not an accident. The incident and fix
are documented in the [v2.10.1 changelog](https://github.com/Abolfazlrwm/Machine-Learning/releases/tag/v2.10.1).

## ADR-003: Network-free CI (fake embedder + mock LLM)

**Status:** Accepted

**Context.** CI that downloads sentence-transformers, torch and model weights
on every run is slow, expensive and flaky — and the evaluation numbers change
for no reason between runs.

**Decision.** The unit suites (320+ tests) run against a deterministic fake
embedder and a scripted mock LLM. Real-model behaviour is guarded by contract
tests (dimensionality, prefix handling) and by recorded live captures; the
golden-set evaluation runs offline against committed index artifacts.

**Consequences.** CI is green in minutes and byte-for-byte reproducible. The
risk of real-model drift is mitigated by ADR-002's contract tests rather than
by downloading gigabytes on every push.

## ADR-004: A bounded re-rank stage after fusion

**Status:** Accepted

**Context.** RRF ordering is good but not perfect; cross-encoder re-ranking
of *everything* would multiply latency.

**Decision.** A re-rank stage refines only the top-k fused candidates. In CI
the scorer is a deterministic lexical re-ranker; a neural cross-encoder can be
switched on offline. On the 100-pair golden set the stage lifts hybrid
Hit@1 from 0.81 to 0.92 with eleven fixes and zero regressions
([evaluation](rag/evaluation.md)).

**Consequences.** The quality gain costs a top-k re-score, not a full corpus
scan; the stage is optional and observable in the per-stage traces.

## ADR-005: Prompt-injection guardrails before prompt build

**Status:** Accepted

**Context.** A RAG pipeline pastes retrieved documents into the prompt —
which makes the corpus an untrusted input channel.

**Decision.** A bilingual (Persian/English) prompt-injection detector runs as
a gate *before* prompt construction; on detection the pipeline answers with a
grounded refusal instead of forwarding the payload ([safety](rag/safety.md)).

**Consequences.** Untrusted corpora get a defined safety posture; false
positives are surfaced in the traces rather than silently swallowed.

## ADR-006: Two-layer answer cache (exact TTL + semantic)

**Status:** Accepted

**Context.** Identical and near-identical questions recur; paying LLM cost
and latency each time is wasteful, but naive caching breaks freshness.

**Decision.** Answers are memoised in an exact TTL + LRU cache, with a
paraphrase-aware semantic layer that only hits above a similarity threshold
([serving API](serving/api.md)).

**Consequences.** Repeated questions return instantly; the threshold guards
against confidently-wrong semantic hits, and every cache hit is labelled in
the response traces.

## ADR-007: SSE streaming and per-stage Prometheus metrics

**Status:** Accepted

**Context.** Generation latency is dominated by token production; operators
need to know which stage spent the time.

**Decision.** The serving layer streams answers over SSE and exports
per-stage timings as Prometheus metrics at `/v1/metrics`
([observability](serving/observability.md)); the committed [API Live page](api-live.md)
shows a real captured session, timings included.

**Consequences.** Frontends render tokens as they arrive and dashboards can
alert on stage-level regressions instead of one opaque latency number.

## ADR-008: Honest numbers or no numbers

**Status:** Accepted

**Context.** Portfolios fail credibility when metrics cannot be re-run or
when inconvenient results are hidden.

**Decision.** Every headline number in the README is reproducible from
committed code, seeds and golden sets. The stock project reports its
out-of-sample ROC-AUC of 0.53 — barely better than chance — *and* the
leakage-free walk-forward backtest that produced it, with costs, because the
interesting part is the honest methodology ([stock](stock.md)). The RAG
assistant keeps a published error-analysis page ([error analysis](rag/error-analysis.md)).

**Consequences.** Some numbers look unimpressive; all of them can be
re-derived, and the reasoning behind them is inspectable.

## ADR-009: Raw-NumPy autograd and transformer (TinyGPT)

**Status:** Accepted

**Context.** Calling `nn.Linear` proves familiarity, not understanding.

**Decision.** TinyGPT implements a reverse-mode autograd engine, a
decoder-only transformer and Adam entirely in NumPy — no PyTorch/JAX — with
every primitive verified against numerical gradients (errors ~1e-10). The
249k-parameter char-level model trains in 65 s on CPU, loss 4.89 → 1.22
([TinyGPT](tiny-gpt.md)).

**Consequences.** The framework-free path is slower to write and strictly
test-covered; it also made ADR-010 possible, since a pure-NumPy stack runs
unchanged in the browser.

## ADR-010: Static Pyodide/WASM Space for TinyGPT

**Status:** Accepted

**Context.** Hugging Face gates Docker/Gradio Spaces behind PRO (a 402 was
hit in a real deploy attempt); free hosting should still show the real model,
not a screenshot.

**Decision.** The unchanged NumPy stack runs in-browser via Pyodide/WASM in a
static Space with a Persian RTL UI — the same 249k-param model, executing on
the visitor's machine ([deployment notes](deployment.md)).

**Consequences.** Zero hosting cost, instant cold start, and the model is
genuinely live; in exchange, inference is bounded to what WASM comfortably
runs — acceptable at this model size.

## ADR-011: One FastAPI layer over all model families

**Status:** Accepted

**Context.** Five projects with five ad-hoc demo scripts makes the portfolio
hard to evaluate as a product.

**Decision.** A single `serving/` FastAPI layer exposes the model families
under one contract — RAG, spam, stock — with one OpenAPI schema, one metrics
endpoint and one Docker Compose stack for local runs
([endpoints](serving/api.md)).

**Consequences.** One demo surface to keep green and one deployment path to
document; per-project notebooks remain for exploration, the API is the product
surface.
