# Retrieval

The retrieval layer is the heart of the assistant: three modes share one
index and are compared on the same golden set.

## Modes

| mode | path | notes |
|---|---|---|
| `dense` | embed question → FAISS cosine → MMR | `mmr_lambda=0.7` balances relevance vs diversity |
| `sparse` | BM25 over tokenised chunks | `k1=1.5`, `b=0.75` |
| `hybrid` | BM25 + dense fused by Reciprocal Rank Fusion | `rrf_k=60`; the default |

### Re-rank stage

Both `dense` and `hybrid` can pass their top candidates through a
re-ranker (`RAGConfig.reranker` / `RAG_RERANKER`):

| backend | what it is | when to use |
|---|---|---|
| `lexical` | rarity-weighted query-term coverage, `w(t) = log(1 + N/df(t))` over the candidate batch — deterministic, dependency-free | CI, offline, smoke deployments |
| `cross-encoder` | `sentence_transformers.CrossEncoder`, e.g. `BAAI/bge-reranker-v2-m3` | production, when the model can be downloaded |
| `noop` | keeps order | tests |

A cross-encoder reads (query, candidate) *jointly* and is more precise
than bi-encoder similarity, but cannot be pre-computed — hence the
two-stage shape: recall with the cheap retriever, precision with the
re-ranker. The lexical stand-in scores the same interface, so swapping is
a config change, not a code change.

### Query understanding

`src/query.py` folds Persian orthography variants, detects mixed-script
questions and appends cross-language expansion terms to the **lexical**
sub-query only — the dense sub-query embeds the visitor's original
wording, so expansions never pull the embedding off-topic.

## Measured quality (CI baseline)

The [CI-reproducible baseline](https://github.com/Abolfazlrwm/Machine-Learning/blob/main/rag-farsi-assistant/evaluation/results/eval_report_ci.md)
uses the hashing embedder so it runs anywhere; the committed E5 numbers
live in `eval_report.md`. On the baseline, the re-rank stage lifts hybrid
ranking to the BM25 ceiling while keeping both fused signals:

| metric | dense | sparse | hybrid | hybrid+rerank |
|---|---|---|---|---|
| hit@1 | 0.70 | **0.90** | 0.80 | **0.90** |
| hit@2 | 0.90 | **0.9667** | 0.9667 | **0.9667** |
| hit@3 | 0.90 | **1.00** | 1.00 | **1.00** |
| mrr | 0.80 | **0.9444** | 0.8944 | **0.9444** |
| ndcg@3 | 0.8262 | **0.9587** | 0.9218 | **0.9587** |

Reproduce it:

```bash
cd rag-farsi-assistant
python evaluate.py --backend fake --compare --out evaluation/results/eval_report_ci
```
