# RAG assistant — architecture

`rag-farsi-assistant/` answers questions **in Persian**, strictly grounded
in a small document corpus, with numbered citations and an explicit
*“I can't answer from these documents”* fallback. Every component is
injectable, so tests and the evaluation harness can swap backends without
touching application code.

## Data flow

```
data/*.txt ──► chunker ──► embedder ──► FAISS store ──┐
                                                      ├─► retriever (MMR / RRF / rerank)
question ─► query analysis (expansion, folding) ──────┘        │
                                                               ▼
                                               numbered context prompt
                                                               │
                                        LLM provider (Ollama / OpenAI-compatible)
                                                               │
                                                               ▼
                                     grounded Persian answer + sources [1][2]
```

## Layers

| layer | module | responsibility |
|---|---|---|
| config | `src/config.py` | dataclass + `RAG_*` env mapping, one source of truth |
| ingestion | `src/data_loader.py`, `src/chunker.py` | load `data/*.txt`, sliding-window chunks (700/150) |
| embeddings | `src/embedder.py` | `sentence-transformers` (real) or deterministic hashing (offline/CI) |
| lexical | `src/bm25.py` | BM25 with k1/b tuning, Persian-aware tokenizer |
| fusion | `src/hybrid.py` | RRF fusion of dense + BM25, MMR diversification, re-rank hook |
| re-rank | `src/rerank.py` | `LexicalReranker` (offline) / `CrossEncoderReranker` (neural) / noop |
| query | `src/query.py` | normalisation, Persian↔English expansion terms, mixed-script flag |
| memory | `src/cache.py`, `src/semantic_cache.py` | TTL/LRU exact cache; embedding-similarity cache for paraphrases |
| safety | `src/guardrails.py` | prompt-injection screening, `off` / `flag` / `block` policies |
| generation | `src/llm/` | Ollama (local, free) or any OpenAI-compatible endpoint |
| serving | `../serving/` | FastAPI wrapper: REST + SSE, auth, Prometheus metrics |

## Design decisions

**Grounding over generation.** The system prompt forces the model to cite
the numbered context chunks and refuse when they do not contain the
answer. The extractive fallback (used when no LLM is configured) returns
the retrieved chunks verbatim — so the whole pipeline is testable without
any model download.

**Provider abstraction.** Switching from Ollama to OpenAI/OpenRouter/Groq
is one environment variable (`RAG_PROVIDER`, `RAG_OPENAI_*`). The heavy
`sentence-transformers` import is lazy: the fake hashing embedder keeps
CI fast and deterministic.

**Determinism for evaluation.** Caches default off, the fake backend is
stable across runs, and the retrieval evaluation therefore produces
byte-identical reports in CI — see [Evaluation](evaluation.md).
