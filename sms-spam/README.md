# SMS Spam Classifier

Baseline NLP classifier for the classic [SMS Spam Collection]
(https://archive.ics.uci.edu/dataset/228/sms+spam+collection): TF-IDF
(1–2 grams) + Logistic Regression wrapped in a single scikit-learn
`Pipeline`, tuned with 5-fold `GridSearchCV`.

## Results

Trained on the included `spam.csv` (5,574 messages, 20% holdout):

| Metric | Score |
|---|---|
| Accuracy (holdout) | **0.975** |
| ROC-AUC | **0.987** |
| Cross-validation accuracy (5-fold) | 0.979 |
| Spam precision / recall / F1 | 0.97 / 0.84 / 0.90 |

Best hyperparameters found by the grid search: `C = 10`, `liblinear` solver.

## Usage

```bash
pip install pandas scikit-learn nltk joblib

# train, evaluate and save the model
python spam_classifier.py --train spam.csv

# classify a single message with the saved model
python spam_classifier.py --predict "Free entry in 2 a wkly comp!"
```

## Pipeline

1. Cleaning — lowercase, strip URLs / digits / punctuation, NLTK stop-word
   removal + WordNet lemmatization.
2. Vectorization — TF-IDF, max 5,000 features, unigrams + bigrams.
3. Model — Logistic Regression, `C` tuned over {0.1, 1, 10}.
4. Persistence — fitted pipeline saved to `spam_classifier.pkl` and reused
   by `--predict`.

Spam recall (0.84) is the weak spot: with ~87% ham messages the model is
conservative, so class weighting or threshold tuning would be the natural
next step.
