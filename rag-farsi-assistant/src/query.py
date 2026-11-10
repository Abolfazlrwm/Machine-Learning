"""Bilingual (Persian <-> English) query understanding for the RAG assistant.

Why this exists
---------------
The hybrid retriever has two very different "ears":

* the **dense** ear (E5 embeddings) already understands that «بیش‌برازش» and
  ``overfitting`` mean the same thing — multilingual models align the two;
* the **sparse** ear (BM25) only sees *tokens*, so a visitor who asks
  ``«overfitting چیست؟»`` gets zero lexical overlap with a corpus that writes
  «بیش‌برازش» — and vice versa.

Farsi technical writing mixes scripts constantly, so before retrieval runs
the question passes through :func:`analyse_query`, which

1. **folds** the text into a canonical form (NFC, Arabic->Persian characters,
   zero-width non-joiner removed, casefolded Latin) — the same rules the
   BM25 tokenizer applies, so comparisons are apples-to-apples;
2. measures the **script mix** (Persian vs Latin letter ratio) — useful
   metadata and the trigger for the ``is_mixed`` flag surfaced by the API;
3. **expands** the query with the cross-language counterparts of the terms
   it recognises, using a small curated ML glossary.

Important design decision: the expansion is applied **only to the lexical
side**. The dense sub-query keeps the visitor's original wording so the
evaluated E5 behaviour (golden set Hit@1 0.967 / MRR 0.983) is unchanged —
expansion strictly adds recall to BM25, it never perturbs the embeddings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .bm25 import normalize_fa

_FA_CHAR = re.compile(r"[\u0600-\u06FF]")
_EN_CHAR = re.compile(r"[A-Za-z]")

# Curated bilingual glossary of the corpus domain. Keys and values are
# stored in natural orthography and folded once at import time.
_GLOSSARY_RAW: dict[str, tuple[str, ...]] = {
    "بیش‌برازش": ("overfitting",),
    "overfitting": ("بیش‌برازش",),
    "کم‌برازش": ("underfitting",),
    "underfitting": ("کم‌برازش",),
    "رانش داده": ("data drift",),
    "data drift": ("رانش داده",),
    "یادگیری ماشین": ("machine learning",),
    "machine learning": ("یادگیری ماشین",),
    "یادگیری با نظارت": ("supervised learning",),
    "supervised learning": ("یادگیری با نظارت",),
    "یادگیری عمیق": ("deep learning",),
    "deep learning": ("یادگیری عمیق",),
    "شبکه عصبی": ("neural network",),
    "neural network": ("شبکه عصبی",),
    "پس‌انتشار": ("backpropagation",),
    "backpropagation": ("پس‌انتشار",),
    "مکانیزم توجه": ("attention",),
    "attention": ("مکانیزم توجه",),
    "ترنسفورمر": ("transformer",),
    "transformer": ("ترنسفورمر",),
    "بردارسازی": ("embedding",),
    "embedding": ("بردارسازی",),
    "مدل زبانی بزرگ": ("llm",),
    "llm": ("مدل زبانی بزرگ",),
    "فیلترینگ مشارکتی": ("collaborative filtering",),
    "collaborative filtering": ("فیلترینگ مشارکتی",),
    "منظم‌سازی": ("regularization",),
    "regularization": ("منظم‌سازی",),
    "اعتبارسنجی متقاطع": ("cross validation",),
    "cross validation": ("اعتبارسنجی متقاطع",),
    "هرزنامه": ("spam",),
    "spam": ("هرزنامه",),
    "بازیابی ترکیبی": ("hybrid retrieval",),
    "hybrid retrieval": ("بازیابی ترکیبی",),
}

# Hard cap so a pathological query cannot balloon into a paragraph.
MAX_EXPANSIONS = 6


def fold(text: str) -> str:
    """Canonical comparison form: NFC, Arabic->Persian chars, no ZWNJ, casefold."""
    text = unicodedata.normalize("NFC", text)
    text = normalize_fa(text)  # also strips the zero-width non-joiner
    return text.casefold()


# Lookup-ready glossary: everything folded once at import time so queries
# are compared in the exact canonical form the BM25 tokenizer produces.
GLOSSARY: dict[str, tuple[str, ...]] = {
    fold(key): tuple(sorted({fold(alt) for alt in alts}))
    for key, alts in _GLOSSARY_RAW.items()
}


def script_ratios(text: str) -> tuple[float, float]:
    """Fraction of alphabetic characters in Persian vs Latin script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0, 0.0
    fa = sum(1 for ch in letters if _FA_CHAR.match(ch))
    en = sum(1 for ch in letters if _EN_CHAR.match(ch))
    n = len(letters)
    return fa / n, en / n


def _contains(haystack: str, needle: str) -> bool:
    """Word-boundary containment (substring for multi-word needles)."""
    if " " in needle:
        return needle in haystack
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


@dataclass
class QueryPlan:
    """Everything the pipeline needs to know about one question."""

    original: str                                   # exactly what the user typed
    normalised: str = ""                            # folded, whitespace-collapsed
    expansions: list[str] = field(default_factory=list)
    expanded: str = ""                              # normalised + expansion terms
    fa_ratio: float = 0.0
    en_ratio: float = 0.0

    @property
    def is_mixed(self) -> bool:
        """True when the query meaningfully mixes Persian and Latin script."""
        return self.fa_ratio > 0.05 and self.en_ratio > 0.05


def analyse_query(text: str) -> QueryPlan:
    """Fold, profile and expand one user question (pure function, no I/O)."""
    normalised = " ".join(fold(text).split())
    fa_ratio, en_ratio = script_ratios(text)

    additions: list[str] = []
    for key in sorted(GLOSSARY):  # deterministic order for tests & logs
        if not _contains(normalised, key):
            continue
        for alt in GLOSSARY[key]:
            alt_folded = fold(alt)
            if alt_folded in additions or _contains(normalised, alt_folded):
                continue
            additions.append(alt_folded)
            if len(additions) >= MAX_EXPANSIONS:
                break
        if len(additions) >= MAX_EXPANSIONS:
            break

    expanded = normalised if not additions else f"{normalised} {' '.join(additions)}"
    return QueryPlan(
        original=text,
        normalised=normalised,
        expansions=additions,
        expanded=expanded,
        fa_ratio=round(fa_ratio, 4),
        en_ratio=round(en_ratio, 4),
    )
