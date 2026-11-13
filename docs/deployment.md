# Deployment

Three deployment shapes, from zero-dependency to full stack.

## 1. Browser-only static demo (zero infra)

The RAG stack is compiled to static assets (`hf-space-static/`) and runs
**entirely in the browser** on Hugging Face Spaces: pre-computed
embeddings, client-side cosine search fused with a BM25 port, RRF fusion
and the grounded answer renderer. No server, no keys, no cold starts.

**Live:** <https://abol0pati-farsi-rag-assistant.static.hf.space/>

Rebuild the embeddings bundle after changing the corpus:

```bash
python scripts/make_static_embeddings.py   # → hf-space-static/embeddings.json
```

## 2. Docker (API + Prometheus + Grafana)

```bash
cd serving
docker compose up --build
```

One compose file brings up the FastAPI service, Prometheus scraping
`/metrics`, and Grafana with a provisioned dashboard — see
[Observability](serving/observability.md).

## 3. Local / Streamlit

```bash
cd rag-farsi-assistant
pip install -r requirements.txt
python main.py build          # chunk → embed → persist index
streamlit run app.py          # chat UI with source panels
```

Swap the LLM provider with env vars: `RAG_PROVIDER=ollama` (local, free)
or `RAG_PROVIDER=openai` with any OpenAI-compatible endpoint
(OpenAI / OpenRouter / Groq).

## CI quality gates

| workflow | jobs |
|---|---|
| `Tests` | RAG suite on Python 3.12 **and** 3.13 (coverage reported), stock pipeline suite, serving API suite |
| `Security` | `bandit` static analysis + `pip-audit` dependency audit, both blocking |

The docs site deploys automatically on every push to `main` that touches
`docs/` or `mkdocs.yml` (GitHub Pages via `actions/deploy-pages`).
