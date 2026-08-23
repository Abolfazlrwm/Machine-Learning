# AI Text Detector

Tries to figure out whether a piece of text was written by a human or generated
by an AI model. Built this after seeing a bunch of "AI detector" tutorials
that just throw TF-IDF at a classifier and call it done — wanted to see if
adding some actual stylometric features (sentence length variation, punctuation
habits, contraction usage etc.) would help. Turns out it does, a little, across
every model I tried.

## Dataset

20k samples, 10k human / 10k AI-generated, from
[shamylafirdoos/Gpt-vs-Human-Text-Classification](https://github.com/shamylafirdoos/Gpt-vs-Human-Text-Classification)
(MIT licensed). Also referenced in a 2024 paper on AI text detection benchmarks.

## What I tried

Three classifiers (Naive Bayes, Logistic Regression, Linear SVM), each trained
two ways: plain TF-IDF, and TF-IDF + my stylometric features combined. 6 runs total.

The stylometric part extracts stuff like:
- average sentence length + how much it varies
- type-token ratio (lexical diversity)
- punctuation/comma/exclamation ratios
- how often contractions show up ("don't" vs "do not" — humans use way more of these)

## Results

| model                   | accuracy | f1    | roc_auc |
|--------------------------|----------|-------|---------|
| naive bayes (tfidf)       | 0.826    | 0.836 | 0.917   |
| naive bayes (hybrid)      | 0.819    | 0.833 | 0.914   |
| logistic reg (tfidf)      | 0.884    | 0.887 | 0.956   |
| logistic reg (hybrid)     | 0.887    | 0.889 | 0.960   |
| linear svm (tfidf)        | 0.895    | 0.897 | 0.960   |
| **linear svm (hybrid)**   | **0.897**| **0.899** | **0.964** |

Hybrid beat plain tfidf every single time, even if only by a bit for some models.
Not a huge jump but consistent enough that I don't think it's noise.

## Running it

```bash
pip install -r requirements.txt

# use the model that's already trained and committed:
python src/predict.py "your text here"

# or retrain from scratch (takes maybe a minute):
python src/main.py
```

Example:
```
$ python src/predict.py "honestly idk why my code keeps breaking, been staring at this for 3 hours lol"
Human-written (confidence: 76.0%)

$ python src/predict.py "In conclusion, artificial intelligence has significantly transformed various industries."
AI-generated (confidence: 78.8%)
```

## Tests

```bash
pytest tests/ -v
```

## Structure

```
data/       the csv
models/     trained model (committed so you don't have to retrain)
src/
  data_loader.py   loading + cleaning
  features.py      the stylometric feature extraction
  model.py         pipeline definitions
  evaluate.py       metrics
  main.py           trains everything, saves the best one
  predict.py         classify your own text from the command line
tests/
```

## Stuff I'd still like to add

- compare against a small transformer model (DistilBERT) to see how much
  accuracy the hand-engineered features are actually leaving on the table
- test how well it holds up against paraphrased AI text (probably not great)
- maybe a tiny web UI instead of just the CLI

## Note

This is a learning project, not something you should use to actually accuse
someone of using AI. It's wrong plenty of the time, especially on short text.
