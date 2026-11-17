"""Text cleaning: lowercase, strip URLs / digits / punctuation, drop NLTK
stop words, WordNet-lemmatize what is left.

The NLTK data (stopwords + wordnet) is fetched lazily on first use —
importing this module must never hit the network.
"""
from __future__ import annotations

import re
import string

_lemmatizer = None
_stop_words: set[str] | None = None


def ensure_nltk() -> None:
    """Download the two small NLTK datasets once, quietly and idempotently."""
    global _lemmatizer, _stop_words
    if _lemmatizer is not None:
        return
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for pkg in ("stopwords", "wordnet"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Lowercase, strip URLs / digits / punctuation, drop stop words,
    lemmatize — the exact sequence the original script used."""
    ensure_nltk()
    assert _lemmatizer is not None and _stop_words is not None  # for type checkers
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [ _lemmatizer.lemmatize(w) for w in text.split() if w not in _stop_words ]
    return " ".join(words)
