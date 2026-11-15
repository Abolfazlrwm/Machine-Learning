# Error analysis

Aggregate metrics hide *where* the pipeline fails. This page is the
written companion of the pre-executed
[`error_analysis.ipynb` notebook](https://github.com/Abolfazlrwm/Machine-Learning/blob/main/rag-farsi-assistant/evaluation/notebooks/error_analysis.ipynb):
every number below is computed from the committed per-question artifact
(`evaluation/results/eval_rows.json`) on the CI-reproducible baseline —
fake hash embedder, BM25 + RRF fusion, lexical re-ranker — so it
regenerates byte-for-byte anywhere.

## Headline numbers (100-pair golden set)

| mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| dense | 0.63 | 0.86 | 0.7350 |
| sparse (BM25) | 0.92 | 1.00 | 0.9583 |
| hybrid (RRF) | 0.81 | 0.99 | 0.8883 |
| **hybrid + rerank** | **0.92** | 0.99 | 0.9517 |

## Failure families

### 1. RRF dilution when the dense channel is silent — the one top-3 miss

«کدام خانواده از مدل‌های توصیه‌گر برنده مسابقه نتفلیکس شد؟» (expected:
doc 05). BM25 ranks the expected document **1st**, but the fake-hash dense
channel produces no signal for it; a document ranked 1st *only* by dense
plus mid-ranked by BM25 accumulates more fused score and the expected
document falls out of the RRF top-3 — so even the re-ranker, which only
sees the fused candidate pool, cannot rescue it.

Fix class: raise `candidate_multiplier` (a larger fused pool) or weight
the sparse channel higher in the fusion. Both are config changes, not
code changes.

### 2. Cross-document topic overlap — the dominant rank-2 cause

Eight questions land at rank 2–3 with the expected document still inside
top-3, so the *answer a user sees is correct*; only the ranking order
slips. The clearest real cases:

* «شبکه کانولوشنی چطور ویژگی‌های تصویر را استخراج می‌کند؟» — CNN is taught
  in **both** doc 02 (architecture) and doc 04 (vision); the ranking must
  pick one.
* «الگوهای رایج استقرار مدل یادگیری ماشین کدام‌اند؟» — the word «الگو» is a
  lexical false friend: doc 01's «از مثال‌ها الگو استخراج می‌کند» shares the
  token.

This is ambiguity, not retrieval failure — and it is exactly the case a
**neural cross-encoder** is designed for: `RAG_RERANKER=cross-encoder`
(`src/rerank.py`) reads query and chunk *jointly* instead of counting
shared tokens.

### 3. Lexical gaps where the dense channel wins

Three questions phrase the concept with synonyms the corpus never uses
(e.g. «فرایند دو مرحله‌ای RAG» — the corpus says «ابتدا اسناد … بازیابی
می‌شود و سپس …»). Dense retrieval bridges part of that gap; the bilingual
query expansion (`src/query.py`) covers the Persian↔English part.

## The re-rank transition: 11 fixes, 0 regressions

Comparing hybrid vs hybrid+rerank per question: the rarity-weighted
lexical re-ranker promoted the expected document to rank 1 in **11**
questions and demoted a correct rank-1 in **zero**. At this corpus scale
that is a pure +0.11 hit@1 gain — the strongest argument for keeping the
re-rank stage on by default in serving.

## Generation quality (extractive mode)

| metric | mean | note |
|---|---|---|
| `context_precision` | 0.7683 | listing-type questions compress several chunks into one gold sentence — token overlap per chunk is low |
| `context_recall` | 0.9337 | worst row 0.22 («یادگیری تقویتی چگونه کار می‌کند؟» — its gold answer rewords the corpus) |
| `faithfulness` | 1.0000 | extractive answers quote chunks by construction |
| `answer_relevancy` | 0.3553 | token-F1 fallback; a real LLM answer scores differently |

A live LLM provider moves these numbers; the extractive run is the
grounding harness, not the ceiling.

## Reproduce

```bash
cd rag-farsi-assistant

# aggregates + RAGAS-style generation suite -> eval_report_ci.{md,json}
python evaluate.py --backend fake --compare --generation \
    --out evaluation/results/eval_report_ci

# per-question rows -> evaluation/results/eval_rows.json
python evaluation/build_eval_rows.py
```

The notebook re-runs as-is (it only reads the committed JSON artifacts).
