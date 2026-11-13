# Machine-Learning Portfolio

A resume-grade machine-learning portfolio built in the open: three real
projects, one repository, with the engineering discipline most repos skip —
**hybrid retrieval with honest evaluation, a production-shaped serving
layer, observability, safety guardrails and reproducible CI**.

## Projects

| project | what it does | signature numbers |
|---|---|---|
| [rag-farsi-assistant](rag/architecture.md) | Persian retrieval-augmented generation, grounded answers with citations | hybrid Hit@1 **0.9667** · MRR **0.9833** (E5, 30 QA pairs) |
| [stock-movement-ml](stock.md) | next-day direction prediction that survives walk-forward validation | out-of-sample backtest with transaction costs |
| [serving](serving/api.md) | one FastAPI service exposing all three via versioned REST + SSE | Prometheus metrics · Docker · CI-gated |

## Highlights

- **Hybrid retrieval** — multilingual E5 dense search fused with BM25 via
  Reciprocal Rank Fusion, MMR re-ranking, query expansion and an optional
  re-rank stage (lexical in CI, neural cross-encoder in production).
- **Honest evaluation** — a golden QA set, RAGAS-style generation metrics
  implemented deterministically, and a CI-reproducible baseline report.
- **Safety** — bilingual prompt-injection guardrails (OWASP LLM01) with
  `flag` / `block` policies surfaced in the API meta block.
- **Observability** — request tracing, Prometheus histograms for cache
  outcomes and stage latency, a provisioned Grafana dashboard.
- **Live demo** — a browser-only static deployment of the RAG stack runs on
  Hugging Face Spaces: [try it here](https://abol0pati-farsi-rag-assistant.static.hf.space/).

## Quickstart

```bash
git clone https://github.com/Abolfazlrwm/Machine-Learning.git
cd Machine-Learning/rag-farsi-assistant

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline, dependency-free (hashing embedder, no model download)
RAG_EMBEDDING_BACKEND=fake python main.py build
RAG_EMBEDDING_BACKEND=fake python app.py

# real embeddings
python main.py build
streamlit run app.py
```

## Where to read next

1. [RAG architecture](rag/architecture.md) — how a question becomes a
   grounded, cited answer.
2. [Evaluation](rag/evaluation.md) — what is measured and why the numbers
   can be reproduced in CI.
3. [Serving API](serving/api.md) — the contract the demo and Docker
   deployment speak.
