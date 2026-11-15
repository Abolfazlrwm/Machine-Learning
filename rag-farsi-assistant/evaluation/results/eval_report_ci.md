# Retrieval & Generation Evaluation

Backend: `fake` · corpus: `data/` · golden set: `100 QA pairs`

## Retrieval modes compared (Hit@k / MRR / nDCG@k)

| metric | dense | sparse | hybrid | hybrid+rerank | best |
|---|---|---|---|---|---|
| hit@1 | 0.63 | **0.92** | 0.81 | 0.92 | sparse |
| hit@2 | 0.8 | **0.99** | 0.92 | 0.97 | sparse |
| hit@3 | 0.86 | **1.0** | 0.99 | 0.99 | sparse |
| mrr | 0.735 | **0.9583** | 0.8883 | 0.9517 | sparse |
| ndcg@3 | 0.7673 | **0.9692** | 0.9144 | 0.9615 | sparse |

`hybrid+rerank` = hybrid results re-ordered by the lexical re-ranker (`src/rerank.py`), same index, no extra retrieval.

## RAGAS-style generation metrics (extractive answer mode)

Answers are produced by the deterministic extractive fallback, so
`faithfulness` doubles as an end-to-end grounding smoke check; run
with a live LLM provider for true generative scoring.

```json
{
  "n_questions": 100,
  "doc_hit_rate": 1.0,
  "context_precision": 0.7683,
  "context_recall": 0.9337,
  "faithfulness": 1.0,
  "answer_relevancy": 0.3553
}
```
