import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import StylometricFeatures, FEATURE_NAMES
from model import tfidf_pipeline, hybrid_pipeline
from sklearn.naive_bayes import MultinomialNB


def test_feature_shape():
    f = StylometricFeatures()
    out = f.transform(["Hello world. This is a test!", "Short."])
    assert out.shape == (2, len(FEATURE_NAMES))


def test_features_dont_break_on_weird_input():
    # was getting nan/inf errors on empty strings earlier so added this check
    f = StylometricFeatures()
    out = f.transform(["", "word", "a b c d e f"])
    assert np.all(np.isfinite(out))


def test_ttr_is_between_0_and_1():
    f = StylometricFeatures()
    out = f.transform(["the the the the the"])
    idx = FEATURE_NAMES.index("ttr")
    assert 0 <= out[0, idx] <= 1


@pytest.fixture
def toy_data():
    human = [
        "honestly i have no idea whats going on with this bug lol",
        "gonna grab lunch, be back in like 20 min",
        "wait so did the meeting actually happen or not??",
    ] * 10
    ai = [
        "In summary, this approach demonstrates significant improvements in overall efficiency.",
        "Furthermore, it is essential to consider the broader implications of this methodology.",
        "Overall, the results indicate a substantial and statistically meaningful enhancement.",
    ] * 10
    texts = human + ai
    labels = [0] * len(human) + [1] * len(ai)
    return pd.Series(texts), pd.Series(labels)


def test_tfidf_pipeline_works(toy_data):
    X, y = toy_data
    pipe = tfidf_pipeline(MultinomialNB())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_hybrid_pipeline_works(toy_data):
    X, y = toy_data
    pipe = hybrid_pipeline(MultinomialNB())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)


def test_hybrid_actually_learns_something(toy_data):
    # these two styles are super different so the model should
    # do way better than random guessing on its own training data
    X, y = toy_data
    pipe = hybrid_pipeline(MultinomialNB())
    pipe.fit(X, y)
    acc = (pipe.predict(X) == y).mean()
    assert acc > 0.8
