# SMS Spam Classifier

Baseline NLP classifier for the classic [SMS Spam Collection]
(https://archive.ics.uci.edu/dataset/228/sms+spam+collection): TF-IDF
(1–2 grams) + Logistic Regression in a single scikit-learn `Pipeline`,
tuned with 5-fold `GridSearchCV`.

## Results

Trained on the bundled `data/spam.csv` (5,574 messages, 20% holdout):

| Metric | Score |
|---|---|
| Accuracy (holdout) | **0.975** |
| ROC-AUC | **0.987** |
| Cross-validation accuracy (5-fold) | 0.979 |
| Spam precision / recall / F1 | 0.97 / 0.84 / 0.90 |

Best hyperparameters found by the grid search: `C = 10`, `liblinear` solver.

## Usage

```bash
pip install -r requirements.txt

# classify a message with the committed artifact
python -m src.predict --predict "Free entry in 2 a wkly comp!"

# retrain, evaluate and refresh artifacts/spam_classifier.pkl
python -m src.train --train data/spam.csv
```

## Structure

```
src/preprocess.py   cleaning: lowercase, strip URLs/digits/punctuation,
                    NLTK stop words + WordNet lemmatization (lazy download)
src/train.py        dataset loader, pipeline builder, GridSearchCV entry
                    point — train_model() returns its metrics dict
src/predict.py      inference against artifacts/spam_classifier.pkl
data/spam.csv       the bundled UCI dataset (5,574 messages)
artifacts/          committed fitted pipeline (~228 KB)
tests/              7 network-free unit tests (cleaning, loader, pipeline,
                    train→predict roundtrip, graceful missing-model path)
```

## Pipeline

1. Cleaning — lowercase, strip URLs / digits / punctuation, NLTK stop-word
   removal + WordNet lemmatization.
2. Vectorization — TF-IDF, max 5,000 features, unigrams + bigrams.
3. Model — Logistic Regression, `C` tuned over {0.1, 1, 10}.
4. Persistence — fitted pipeline saved to `artifacts/spam_classifier.pkl`
   and reused by `predict.py`.

Spam recall (0.84) is the weak spot: with ~87% ham messages the model is
conservative, so class weighting or threshold tuning would be the natural
next step.
