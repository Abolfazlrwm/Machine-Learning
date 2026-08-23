from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler

from features import StylometricFeatures


def tfidf_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def hybrid_pipeline(clf):
    # combines tfidf with the stylometric stuff from features.py
    combined = FeatureUnion([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("stylo", Pipeline([
            ("extract", StylometricFeatures()),
            ("scale", MaxAbsScaler()),
        ])),
    ])
    return Pipeline([
        ("features", combined),
        ("clf", clf),
    ])


# tried a few different classifiers, keeping all 3 to compare
MODELS = {
    "naive_bayes": lambda: MultinomialNB(),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "linear_svm": lambda: SGDClassifier(loss="hinge", max_iter=1000, random_state=42),
}
