"""BM25 (Okapi) sparse retriever implemented from scratch — zero dependencies.

BM25 scores a query against documents with a TF-IDF-style formula that
saturates term frequency and length-normalises documents:

    score(q, d) = Σ_{t ∈ q} IDF(t) · tf(t,d) · (k1 + 1)
                  / (tf(t,d) + k1 · (1 - b + b · |d| / avgdl))

    IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

The dense (embedding) retriever understands *meaning*; BM25 understands
*exact vocabulary*. Farsi technical text is full of terms where exact
lexical match matters — «پس‌انتشار», «فیلترینگ مشارکتی», RAG, BERT — which
is exactly what motivates the hybrid retriever in ``src.hybrid``.

Persian-specific normalisation makes matching robust on the Farsi corpus:

* Arabic Yeh/Keh (ي ك) are mapped to the Persian forms (ی ک);
* the zero-width non-joiner (U+200C) is removed so «می‌شود» and «میشود»
  produce the same token;
* punctuation is stripped, latin letters and digits are kept (RAG, GPT, 3D);
* a small built-in Persian stopword list drops high-frequency glue words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Persian-aware tokenisation
# ---------------------------------------------------------------------------

_CHAR_MAP = str.maketrans({
    "ي": "ی",   # Arabic Yeh  -> Persian Yeh
    "ك": "ک",   # Arabic Kaf  -> Persian Kaf
    "ة": "ه",
    "أ": "ا",
    "إ": "ا",
    "\u200c": "",  # zero-width non-joiner
})

# Anything that is not a letter/digit/underscore (unicode) or whitespace.
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

# Conservative stopword list: function words only, no content-bearing stems.
PERSIAN_STOPWORDS = frozenset({
    "از", "به", "با", "که", "را", "و", "در", "است", "این", "آن", "برای",
    "تا", "هم", "بر", "یا", "اگر", "ولی", "اما", "هر", "می", "های", "شود",
    "شد", "کند", "کرد", "نیز", "یک", "چون", "روی", "بین", "همان", "هیچ",
})


def normalize_fa(text: str) -> str:
    """Apply the Persian character normalisations described above."""
    return text.translate(_CHAR_MAP)


def tokenize(text: str) -> list[str]:
    """Lowercase, normalise, strip punctuation and stopwords -> token list."""
    text = normalize_fa(text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return [
        tok for tok in _WHITESPACE_RE.split(text)
        if tok and tok not in PERSIAN_STOPWORDS
    ]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------


class BM25Okapi:
    """In-memory BM25 index over a list of documents (chunk texts).

    Parameters mirror the classic Okapi constants:

    * ``k1`` — term-frequency saturation (higher = less saturation);
    * ``b``  — document-length normalisation strength (0 = none, 1 = full).
    """

    def __init__(
        self,
        corpus: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer=tokenize,
    ):
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._tokenizer = tokenizer

        self._doc_tf: list[Counter[str]] = []
        self._doc_len = np.zeros(len(corpus), dtype="float32")
        df: Counter[str] = Counter()

        for i, text in enumerate(corpus):
            tokens = self._tokenizer(text)
            tf = Counter(tokens)
            self._doc_tf.append(tf)
            self._doc_len[i] = len(tokens)
            df.update(tf.keys())

        self._n_docs = len(corpus)
        self._avgdl = float(self._doc_len.mean()) if self._n_docs else 0.0
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (self._n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def __len__(self) -> int:
        return self._n_docs

    # ---------------------------------------------------------------- scoring
    def scores(self, query: str) -> np.ndarray:
        """BM25 score of *query* against every document (order preserved)."""
        out = np.zeros(self._n_docs, dtype="float64")
        if not self._n_docs:
            return out
        query_terms = self._tokenizer(query)
        if not query_terms or self._avgdl == 0.0:
            return out
        for term in set(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue  # unseen in corpus -> no signal
            for i, tf in enumerate(self._doc_tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * self._doc_len[i] / self._avgdl
                )
                out[i] += idf * freq * (self.k1 + 1.0) / denom
        return out

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Top-``top_k`` documents as ``(index, score)`` sorted by score desc.

        Documents with a score of zero are excluded — an exact-vocabulary
        matcher has no meaningful ranking among them.
        """
        if top_k <= 0 or not self._n_docs:
            return []
        all_scores = self.scores(query)
        nonzero = np.nonzero(all_scores > 0)[0]
        if not len(nonzero):
            return []
        order = nonzero[np.argsort(-all_scores[nonzero])][: int(top_k)]
        return [(int(i), float(all_scores[i])) for i in order]
