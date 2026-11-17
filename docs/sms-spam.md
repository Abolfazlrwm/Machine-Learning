# SMS spam classifier

The portfolio's NLP baseline, refactored into the same structured shape
as the rest of the projects (`src/` + `tests/` + bundled data + committed
artifact): TF-IDF (unigrams + bigrams) → Logistic Regression, tuned with
5-fold `GridSearchCV`.

## Results

5,574 bundled SMS messages, 20% holdout, seed 42 — the numbers are
reproduced exactly by the refactored package (`python -m src.train
--train data/spam.csv`):

| Metric | Score |
|---|---|
| Accuracy (holdout) | **0.975** |
| ROC-AUC | **0.987** |
| Cross-validation accuracy (5-fold) | 0.979 |
| Spam precision / recall / F1 | 0.97 / 0.84 / 0.90 |

Best hyperparameters: `C = 10`, `liblinear` solver.

## Design notes

- **Lazy NLTK.** The original script downloaded stop-word/WordNet data at
  *import* time — a network call on every use. The package fetches the two
  small datasets once, on first `clean_text()` call, idempotently; the CI
  job pre-fetches them in a dedicated step.
- **Metrics as data.** `train_model()` returns a metrics dict (accuracy,
  ROC-AUC, CV accuracy, per-class precision/recall/F1, best params)
  instead of only printing — the README table is generated from it and the
  roundtrip test asserts on it.
- **Committed artifact.** The fitted pipeline (~228 KB) lives at
  `artifacts/spam_classifier.pkl`, so `python -m src.predict` works
  without retraining.
- **Graceful failure.** Predicting without a model raises a
  `FileNotFoundError` that tells you the exact train command; the CLI
  turns it into a clean error exit instead of a traceback.

## Pipeline

1. Cleaning — lowercase, strip URLs / digits / punctuation, NLTK stop-word
   removal + WordNet lemmatization.
2. Vectorization — TF-IDF, max 5,000 features, unigrams + bigrams.
3. Model — Logistic Regression, `C` tuned over {0.1, 1, 10}.
4. Persistence — fitted pipeline saved to `artifacts/spam_classifier.pkl`.

Spam recall (0.84) is the honest weak spot: with ~87% ham messages the
model is conservative — class weighting or threshold tuning would be the
natural next step.

## Testing

Seven deterministic unit tests: cleaning behaviour (noise stripping,
stop-word removal, lemmatization, determinism), dataset loading with
label mapping, pipeline/param-grid shape, a full train → predict
roundtrip on a synthetic separable set, and the missing-model error path.
