"""SMS spam classifier — TF-IDF (1-2 grams) + Logistic Regression.

Structured package: `preprocess` owns the text cleaning (NLTK stopwords +
WordNet lemmatization), `train` owns dataset loading, the sklearn pipeline
and the GridSearchCV entry point, `predict` owns inference against the
committed artifact. The behaviour (and therefore the README numbers) is a
refactor-preserving port of the original single-file script.
"""
