# Persian RAG Assistant 📚

A retrieval-augmented generation (RAG) assistant that answers questions
**in Persian**, strictly grounded in a small document corpus — with source
citations and an explicit "I can't answer from these documents" fallback.

Designed as a clean, dependency-light reference implementation:

- **hybrid retrieval**: **multilingual E5 embeddings** + **FAISS** fused with a
  **from-scratch BM25** via **Reciprocal Rank Fusion**, plus optional
  **cross-encoder re-ranking** and **MMR** diversity,
- **three switchable modes**: `dense` / `sparse` / `hybrid` — one CLI flag or
  env var, evaluated side-by-side in `evaluation/results/`,
- **provider abstraction**: run the LLM locally with **Ollama** (free, no key)
  or switch to **any OpenAI-compatible API** (OpenAI / OpenRouter / Groq) with
  one environment variable,
- **RAGAS-style evaluation suite**: context precision/recall, faithfulness and
  answer relevancy over a 30-pair golden set — deterministic, no judge LLM needed,
- **Streamlit chat UI** with source panels, plus a CLI and an evaluation script,
- a **"fake" hashing embedder** so the whole pipeline runs without model
  downloads — used by the test suite and CI.

> 🔎 **Live demo:** [farsi-rag-assistant on Hugging Face Spaces](https://huggingface.co/spaces/Abol0pati/farsi-rag-assistant)
> — a browser-only edition of the hybrid retriever (E5 + BM25 + RRF, zero
> server-side inference). Source: [`../hf-space-static/`](../hf-space-static/)

## Architecture

```
data/*.txt ──► chunker ──► embedder ──► FAISS store ──┐
                                                      ├─► RRF fusion ─► top-k
question ──► BM25 (from scratch) ─────────────────────┘        │
                                                               ▼
                                               numbered context prompt
                                                               │
                                        LLM provider (Ollama / OpenAI-compatible)
                                                               │
                                                               ▼
                                     grounded Persian answer + sources [1][2]
```

| Module | Responsibility |
|---|---|
| `src/config.py` | single `RAGConfig` dataclass, `RAG_*` env overrides |
| `src/data_loader.py` | loads `.txt` / `.md` corpus into `Document` objects |
| `src/chunker.py` | paragraph→sentence-aware splitting with overlap (Persian sentence enders `؟ ! .` supported) |
| `src/embedder.py` | `SentenceTransformerEmbedder` (auto E5 prefixes) + `FakeHashEmbedder` fallback |
| `src/vector_store.py` | FAISS `IndexFlatIP` (cosine on normalised vectors) with pure-NumPy fallback, save/load |
| `src/retriever.py` | top-k retrieval + Maximal Marginal Relevance for diversity |
| `src/bm25.py` | **BM25 (Okapi) from scratch** — Persian normalisation (Arabic Yeh/Keh, ZWNJ, stopwords), zero dependencies |
| `src/hybrid.py` | **Reciprocal Rank Fusion** + `HybridRetriever` with `dense` / `sparse` / `hybrid` modes |
| `src/rerank.py` | optional **cross-encoder re-ranker** hook (e.g. `BAAI/bge-reranker-v2-m3`) |
| `src/llm/` | `LLMProvider` interface, `OllamaProvider`, `OpenAIProvider`, factory |
| `src/prompts.py` | strict Persian system prompt: answer only from context, cite `[n]`, admit gaps |
| `src/pipeline.py` | `RAGPipeline` facade: build/load index, retrieve, generate |
| `evaluation/metrics.py` | **RAGAS-style suite**: context precision/recall, faithfulness, answer relevancy |
| `app.py` / `main.py` / `evaluate.py` | Streamlit chat / CLI / retrieval + generation metrics |

## Quick start

```bash
pip install -r requirements.txt

# 1) build the vector index from data/
python main.py build

# 2) ask a question (CLI)
python main.py ask "بیش‌برازش چیست و چطور با آن مقابله می‌شود؟"

#    pick the retrieval mode explicitly (default = hybrid)
python main.py ask "پس‌انتشار خطا چگونه کار می‌کند؟" --mode sparse

# 3) or chat in the browser
streamlit run app.py
```

### Choosing the LLM

| Provider | Setup | Cost |
|---|---|---|
| Ollama (default) | install [Ollama](https://ollama.com), then `ollama pull llama3.1 && ollama serve` | free, local |
| OpenAI-compatible | `export OPENAI_API_KEY=...` (optionally `OPENAI_BASE_URL`, `OPENAI_MODEL`) | API pricing |

Switch with `python main.py ask ... --provider openai` or the sidebar radio in
the Streamlit app. Any OpenAI-compatible endpoint works — e.g. OpenRouter:

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct
```

### Embedding backends

| Backend | Model | Use case |
|---|---|---|
| `sentence-transformers` | `intfloat/multilingual-e5-base` (default) | real semantic retrieval, strong Persian quality |
| `sentence-transformers` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | lighter / faster download |
| `fake` | hashed bag-of-words (no downloads) | tests, CI, plumbing demos |

## 🔎 Hybrid retrieval (BM25 + RRF)

Dense embeddings understand *meaning*; BM25 understands *exact vocabulary* —
and Farsi technical text is full of terms where exact lexical match matters
(«پس‌انتشار», «فیلترینگ مشارکتی», RAG, BERT). The hybrid retriever fetches
candidates from both systems and fuses their **rankings** (score scales are
incomparable) with Reciprocal Rank Fusion:

    RRF(d) = Σ_rankers 1 / (rrf_k + rank_ranker(d))

The BM25 index is implemented from scratch in `src/bm25.py` (no rank-bm25
dependency) with Persian-aware tokenisation: Arabic Yeh/Keh → Persian forms,
zero-width non-joiner removal, punctuation stripping and a built-in stopword
list. An optional cross-encoder re-ranker (`src/rerank.py`) can refine the
fused list when `sentence-transformers` is installed.

## 📊 Evaluation

### Retrieval — dense vs sparse vs hybrid

30 golden Persian questions over the 6-document corpus
(`evaluation/qa_pairs.json`, 5 per document), measuring whether the correct
source document appears in the top-k (`Hit@k`) and the mean reciprocal rank
(`MRR`). Measured with **`intfloat/multilingual-e5-base`** embeddings (CPU):

```bash
python evaluate.py --compare --generation --rebuild \
    --out evaluation/results/eval_report
```

| Retrieval mode | Hit@1 | Hit@2 | Hit@3 | MRR |
|---|---|---|---|---|
| dense (E5 + MMR) | 0.933 | **1.000** | **1.000** | 0.967 |
| sparse (BM25 from scratch) | 0.933 | 0.967 | **1.000** | 0.961 |
| hybrid (E5 + BM25 via RRF) | **0.967** | **1.000** | **1.000** | **0.983** |

Dense and sparse each miss a different question — exactly the failure mode
Reciprocal Rank Fusion exists for. Fusing the two rankings recovers both
(Hit@1 0.933 → **0.967**, MRR 0.967 → **0.983**), so `hybrid` is the default
retrieval mode. (With the dependency-free `--backend fake` hash embedder the
picture flips and sparse dominates — a good reminder that fusion is only as
good as the signals you feed it.)

### Generation — RAGAS-style suite (`evaluation/metrics.py`)

Deterministic, judge-free proxies of the four RAGAS dimensions, computed over
the same golden set. Answers are produced by the extractive fallback and
scored with the same E5 embedder (`answer relevancy` = question ↔ answer
cosine), so the whole suite reproduces on CPU; run it behind a live LLM
provider for true generative scoring:

| Metric | Value |
|---|---|
| doc hit rate | 1.000 |
| context precision (AP-style) | 0.942 |
| context recall | 0.979 |
| faithfulness | 1.000 |
| answer relevancy (E5 cosine) | 0.846 |

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q     # 72 tests, no network / no model downloads
```

Covers: chunk size & overlap guarantees, determinism and normalisation of
embeddings, FAISS/numpy store round-trips, MMR de-duplication, prompt
construction (context numbering, strict system prompt), index save/load
equivalence, the provider factory, BM25 ranking + Persian normalisation,
RRF fusion with hand-computed scores, hybrid mode switching, the reranker
plumbing and the RAGAS-style metrics on hand-computed fixtures.
