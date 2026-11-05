# Retrieval & Generation Evaluation

Backend: `fake` · corpus: `data/` · golden set: `30 QA pairs`

## Retrieval modes compared (Hit@k / MRR)

| metric | dense | sparse | hybrid | best |
|---|---|---|---|---|
| hit@1 | 0.7 | **0.9333** | 0.8333 | sparse |
| hit@2 | 0.9 | **0.9667** | 0.9667 | sparse |
| hit@3 | 0.9 | **1.0** | 1.0 | sparse |
| mrr | 0.8 | **0.9611** | 0.9111 | sparse |

## RAGAS-style generation metrics (extractive answer mode)

Answers are produced by the deterministic extractive fallback, so
`faithfulness` doubles as an end-to-end grounding smoke check; run
with a live LLM provider for true generative scoring.

```json
{
  "n_questions": 30,
  "doc_hit_rate": 1.0,
  "context_precision": 0.8167,
  "context_recall": 0.9429,
  "faithfulness": 1.0,
  "answer_relevancy": null
}
```
