# Retrieval & Generation Evaluation

Backend: `fake` · corpus: `data/` · golden set: `30 QA pairs`

## Retrieval modes compared (Hit@k / MRR / nDCG@k)

| metric | dense | sparse | hybrid | hybrid+rerank | best |
|---|---|---|---|---|---|
| hit@1 | 0.7 | **0.9** | 0.8 | 0.9 | sparse |
| hit@2 | 0.9 | **0.9667** | 0.9667 | 0.9667 | sparse |
| hit@3 | 0.9 | **1.0** | 1.0 | 1.0 | sparse |
| mrr | 0.8 | **0.9444** | 0.8944 | 0.9444 | sparse |
| ndcg@3 | 0.8262 | **0.9587** | 0.9218 | 0.9587 | sparse |

`hybrid+rerank` = hybrid results re-ordered by the lexical re-ranker (`src/rerank.py`), same index, no extra retrieval.
