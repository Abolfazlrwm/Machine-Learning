# Retrieval & Generation Evaluation

Backend: `sentence-transformers` · corpus: `data/` · golden set: `30 QA pairs`

## Retrieval modes compared (Hit@k / MRR)

| metric | dense | sparse | hybrid | best |
|---|---|---|---|---|
| hit@1 | 0.9333 | 0.9333 | **0.9667** | hybrid |
| hit@2 | **1.0** | 0.9667 | 1.0 | dense |
| hit@3 | **1.0** | 1.0 | 1.0 | dense |
| mrr | 0.9667 | 0.9611 | **0.9833** | hybrid |

## RAGAS-style generation metrics (extractive answer mode)

Answers are produced by the deterministic extractive fallback, so
`faithfulness` doubles as an end-to-end grounding smoke check; run
with a live LLM provider for true generative scoring.

```json
{
  "n_questions": 30,
  "doc_hit_rate": 1.0,
  "context_precision": 0.9417,
  "context_recall": 0.9788,
  "faithfulness": 1.0,
  "answer_relevancy": 0.8461
}
```
