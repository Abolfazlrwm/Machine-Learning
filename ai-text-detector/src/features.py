import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# messing around with different features here, some of these
# ended up not mattering much but keeping them anyway

SENT_SPLIT = re.compile(r"[.!?]+")
WORD_SPLIT = re.compile(r"\b\w+\b")


def get_stylo_features(text):
    words = WORD_SPLIT.findall(text)
    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]

    if len(words) == 0:
        words = [""]
    if len(sentences) == 0:
        sentences = [""]

    n_words = len(words)
    n_sent = len(sentences)

    word_lens = [len(w) for w in words]
    sent_lens = [len(WORD_SPLIT.findall(s)) for s in sentences]

    avg_word_len = sum(word_lens) / len(word_lens)
    avg_sent_len = sum(sent_lens) / len(sent_lens)
    sent_len_std = np.std(sent_lens) if len(sent_lens) > 1 else 0.0

    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / n_words  # type-token ratio

    punct_count = sum(1 for c in text if c in ".,;:!?")
    punct_ratio = punct_count / max(len(text), 1)

    # contractions like "don't", "it's" - humans use these way more
    contractions = re.findall(r"\b\w+'\w+\b", text)
    contraction_ratio = len(contractions) / n_words

    comma_ratio = text.count(",") / n_words
    exclaim_ratio = text.count("!") / n_sent

    return [
        avg_word_len,
        avg_sent_len,
        sent_len_std,
        ttr,
        punct_ratio,
        contraction_ratio,
        comma_ratio,
        exclaim_ratio,
        n_words,
        n_sent,
    ]


FEATURE_NAMES = [
    "avg_word_len", "avg_sent_len", "sent_len_std", "ttr",
    "punct_ratio", "contraction_ratio", "comma_ratio",
    "exclaim_ratio", "n_words", "n_sent"
]


class StylometricFeatures(BaseEstimator, TransformerMixin):
    # wraps get_stylo_features so it plugs into an sklearn Pipeline

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [get_stylo_features(t) for t in X]
        return np.array(rows)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)
