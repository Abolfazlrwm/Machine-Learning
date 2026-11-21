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

### 2b. The API as a Hugging Face Docker Space

> Hugging Face requires a **PRO subscription** for Docker/Gradio Spaces on
> free cpu-basic hardware (static Spaces remain free). Without PRO the
> create call returns `402 Payment Required`; the package below stays
> ready for when you subscribe, or point any Docker host at
> `hf-space-serving/Dockerfile`.

`hf-space-serving/` packages the same serving stack for Spaces
(`sdk: docker`, port 7860). The image build **pre-bakes the RAG index and
the E5 model cache**, so the Space is warm as soon as it reports Running;
the SMS service ships its committed artifact, and services without
artifacts degrade to a machine-readable `503` instead of crashing. Deploy
with one command (needs a write token):

```bash
HF_TOKEN=hf_... make space-deploy-api
# Swagger UI: https://abol0pati-ml-portfolio-api.hf.space/docs
```

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
