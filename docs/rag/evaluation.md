# Evaluation

Evaluation in this repo is designed to be **reproducible anywhere** —
including CI, where no GPU and no model download is available. That
shapes three choices: deterministic metrics, a deterministic offline
backend, and reports committed next to the code they describe.

## Golden set

`evaluation/qa_pairs.json` holds 30 hand-written Persian questions, each
mapped to the document that must be retrieved (`expected_doc`). Retrieval
quality is computed per question; generation quality runs the full
pipeline over the same set.

## Retrieval metrics

| metric | definition |
|---|---|
| `hit@k` | expected document inside the top-k |
| `mrr` | mean reciprocal rank of the expected document |
| `ndcg@k` | binary-relevance nDCG — one relevant doc, so `1/log2(rank+1)` |

`ndcg@k` sits alongside Hit@k/MRR because it discounts deeper hits the
same way but integrates cleanly with graded relevance later.

## Generation metrics (RAGAS-style, judge-free)

The original RAGAS suite asks an LLM judge to score faithfulness. Here
the same four dimensions are implemented with transparent token-overlap
heuristics, so every score is reproducible and auditable:

| metric | question it answers |
|---|---|
| `context_precision` | are the retrieved chunks on-topic? (AP-style, token-F1 ≥ τ) |
| `context_recall` | how much of the gold answer's vocabulary do the chunks cover? |
| `faithfulness` | what fraction of the answer's sentences are grounded in the contexts? |
| `answer_relevancy` | is the answer close to the question? (embedder cosine or token-F1) |

## Committed reports

| file | backend | purpose |
|---|---|---|
| `evaluation/results/eval_report.md` | `sentence-transformers` (multilingual E5) | the headline numbers: hybrid Hit@1 0.9667, MRR 0.9833 |
| `evaluation/results/eval_report_ci.md` | `fake` (hashing) | the CI-reproducible baseline, including the re-rank column |

## Reproduce

```bash
# full comparison + RAGAS-style suite (needs the E5 model locally)
python evaluate.py --compare --generation --out evaluation/results/report

# the CI baseline — no downloads, byte-stable
python evaluate.py --backend fake --compare --out evaluation/results/eval_report_ci
```

!!! note
    With the extractive fallback (no LLM configured), `faithfulness` doubles
    as an end-to-end grounding smoke check. Point the pipeline at a live
    provider for true generative scoring.
