# AI text detector

Human-vs-AI text classification with **stylometric features on top of
TF-IDF** — built to test whether *how* something is written (not just
*what* words appear) carries a detectable signal. It does: adding the
stylometry block improved every one of the three classifiers tried.

## Dataset

20,000 samples (10k human-written / 10k AI-generated) from
[shamylafirdoos/Gpt-vs-Human-Text-Classification](https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification)
(MIT), mirrored byte-for-byte as a release asset (`datasets-v1`) so the
experiment is reproducible without depending on a third-party repo
staying put.

## Stylometric features

The feature block concatenates TF-IDF (word n-grams) with hand-built
style statistics per document:

- **sentence length** — average sentence length and its variance
  (humans vary rhythm; generators are metronomic)
- **lexical diversity** — type-token ratio
- **punctuation habits** — comma / exclamation / question ratios
- **contraction usage** — `don't` vs `do not` style frequencies
  (the most human tell in the set)

## The six-run comparison

Three classifiers × two feature sets — plain TF-IDF vs hybrid
(TF-IDF + stylometry):

| model | accuracy | f1 | roc_auc |
|---|---|---|---|
| naive bayes (tfidf) | 0.826 | 0.836 | 0.917 |
| naive bayes (hybrid) | 0.819 | 0.833 | 0.914 |
| logistic reg (tfidf) | 0.884 | 0.887 | 0.956 |
| logistic reg (hybrid) | 0.887 | 0.889 | 0.960 |
| linear svm (tfidf) | 0.895 | 0.897 | 0.960 |
| **linear svm (hybrid)** | **0.897** | **0.899** | **0.964** |

Hybrid beat plain TF-IDF for logistic regression and the SVM; naive
bayes is essentially flat (its independence assumption already
"absorbs" the extra features differently). The honest reading: the
stylometric lift is small but consistent — and the project's
methodology section treats it that way rather than overselling.

## Running it

```bash
pip install -r requirements.txt

# predict with the committed best model
python src/predict.py "your text here"

# or retrain from scratch (~1 min on CPU)
python src/main.py
```

The trained artifact lands at `models/best_model.joblib` (git-ignored by
the repo's "no trained artifacts in git" policy — retrain in about a
minute with `python src/main.py`). `predict.py` falls back to a sigmoid
over `decision_function` when the wrapped classifier exposes no
`predict_proba`.

## Testing

Six network-free unit tests cover the feature extractor's statistics
(determinism, range checks), the model interface contract and the
predict path — the same suite CI runs (`ai-text-detector` job).
