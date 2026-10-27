# Persian RAG Assistant 📚

A retrieval-augmented generation (RAG) assistant that answers questions
**in Persian**, strictly grounded in a small document corpus — with source
citations and an explicit "I can't answer from these documents" fallback.

Designed as a clean, dependency-light reference implementation:

- **multilingual E5 embeddings** + **FAISS** vector search with **MMR** re-ranking,
- **provider abstraction**: run the LLM locally with **Ollama** (free, no key)
  or switch to **any OpenAI-compatible API** (OpenAI / OpenRouter / Groq) with
  one environment variable,
- **Streamlit chat UI** with source panels, plus a CLI and a retrieval
  evaluation script,
- a **"fake" hashing embedder** so the whole pipeline runs without model
  downloads — used by the test suite and CI.

## Architecture

```
data/*.txt ──► chunker ──► embedder ──► FAISS store ──┐
                                                      ├─► retriever (MMR)
question ─────────────────────────────────────────────┘        │
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
| `src/llm/` | `LLMProvider` interface, `OllamaProvider`, `OpenAIProvider`, factory |
| `src/prompts.py` | strict Persian system prompt: answer only from context, cite `[n]`, admit gaps |
| `src/pipeline.py` | `RAGPipeline` facade: build/load index, retrieve, generate |
| `app.py` / `main.py` / `evaluate.py` | Streamlit chat / CLI / retrieval metrics |

## Quick start

```bash
pip install -r requirements.txt

# 1) build the vector index from data/
python main.py build

# 2) ask a question (CLI)
python main.py ask "بیش‌برازش چیست و چطور با آن مقابله می‌شود؟"

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

## 📊 Retrieval evaluation

10 golden Persian questions over the 6-document corpus
(`evaluation/qa_pairs.json`), measuring whether the correct source document
appears in the top-k (`Hit@k`) and the mean reciprocal rank (`MRR`):

```bash
python evaluate.py            # builds/loads the index, prints metrics
```

| Embedding model | Hit@1 | Hit@2 | Hit@3 | MRR |
|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (384d) | 0.80 | 0.90 | 1.00 | 0.883 |
| **`intfloat/multilingual-e5-base` (768d, default)** | **0.90** | **1.00** | **1.00** | **0.950** |

The E5 query/passage prefixes are applied automatically by the embedder.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q     # 33 tests, no network / no model downloads
```

Covers: chunk size & overlap guarantees, determinism and normalisation of
embeddings, FAISS/numpy store round-trips, MMR de-duplication, prompt
construction (context numbering, strict system prompt), index save/load
equivalence and the provider factory.
