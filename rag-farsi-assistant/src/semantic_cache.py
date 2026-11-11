"""Paraphrase-aware answer cache: embedding-similarity lookup over answers.

The exact-match :class:`~src.cache.TTLCache` only reuses an answer when the
folded question is byte-identical. Real traffic paraphrases: «بیش‌برازش
چیست؟», «overfitting یعنی چه» and «تعریف بیش‌برازش» are three cache misses
that all want the same answer. The semantic cache closes that gap:

* every cached question is embedded **once** and stored alongside the value;
* a lookup embeds the incoming question and scans the (bounded) entry set
  for the highest cosine similarity — embeddings arrive L2-normalised from
  every backend, so cosine reduces to a dot product;
* ``similarity >= threshold`` is a hit; the best-matching entry is served
  exactly as the exact cache would (same answer, same sources, zero
  retrieval + generation cost). One query embedding replaces the whole
  pipeline, which is the entire point.

Design notes
------------
* **Namespaces.** Cache identity still depends on the settings that change
  the output (``top_k``, retrieval mode, reranker, embedder, index size) —
  the caller passes them as a ``namespace`` so two pipelines sharing one
  cache never cross-contaminate.
* **Exact-vector shortcut.** An identical embedded question hits with
  ``similarity == 1.0`` via byte-equality, so float rounding can never
  turn an exact repeat into a miss — even at ``threshold = 1.0``.
* **TTL + LRU reuse.** Entries carry an injectable clock timestamp; expiry
  is checked lazily during scans and puts, and the insertion order doubles
  as the LRU order, exactly like :class:`~src.cache.TTLCache`.
* **Embedding outside the lock.** Only the dict scan is synchronised, so a
  slow (real-model) ``embed`` call never blocks other lookups longer than
  necessary.
* **Stats as first-class citizens.** ``hits`` / ``misses`` / ``expirations``
  / ``evictions`` / ``hit_rate`` mirror the exact cache, plus the
  similarity observed on the last lookup for tuning the threshold.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from .embedder import BaseEmbedder


def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector defensively (zero vector stays zero)."""
    vec = np.asarray(vec, dtype="float32").ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


class SemanticCache:
    """TTL + LRU cache whose lookup keys are embedding similarities.

    Parameters
    ----------
    embedder:
        Any :class:`~src.embedder.BaseEmbedder`; ``embed([text])`` must
        return unit-norm rows (the contract every backend already keeps).
    threshold:
        Minimum cosine similarity for a hit, in ``[0.0, 1.0]``. 0.92 is a
        sensible default for E5-family models; raise it to trade hit rate
        for precision, lower it to serve more paraphrases.
    maxsize:
        LRU bound on stored answers.
    ttl:
        Seconds an entry stays fresh; ``0`` means "never expires".
    clock:
        Injectable monotonic clock (tests inject a fake one).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.92,
        maxsize: int = 128,
        ttl: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0.0, 1.0]")
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl < 0:
            raise ValueError("ttl must be >= 0")
        self._embedder = embedder
        self._threshold = float(threshold)
        self._maxsize = maxsize
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        # key -> (stored_at, namespace, vector, value); insertion order = LRU
        self._entries: "OrderedDict[str, tuple[float, str, np.ndarray, Any]]" = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0
        self.last_similarity: float | None = None

    # ------------------------------------------------------------- lookup
    def lookup(
        self, question: str, namespace: str = ""
    ) -> tuple[bool, Any, float | None]:
        """Return ``(hit, value, similarity)`` for the best match.

        ``similarity`` is ``1.0`` for an exact embedded repeat, otherwise
        the best cosine found (also returned on a miss, so callers can log
        near-misses while tuning the threshold).
        """
        vec = _unit(self._embedder.embed([question])[0])
        key = self._key(question, namespace)
        now = self._clock()
        with self._lock:
            self._drop_expired(now)
            best_sim = -1.0
            best_item_key: str | None = None
            for item_key, (stored_at, ns, stored_vec, _value) in self._entries.items():
                if ns != namespace:
                    continue
                if np.array_equal(vec, stored_vec):
                    sim = 1.0  # byte-equal vectors: immune to float rounding
                else:
                    sim = min(float(np.dot(vec, stored_vec)), 1.0)
                if sim > best_sim:
                    best_sim, best_item_key = sim, item_key
            self.last_similarity = best_sim if best_sim >= 0.0 else None
            if best_item_key is not None and best_sim >= self._threshold:
                self._entries.move_to_end(best_item_key)
                self._hits += 1
                return True, self._entries[best_item_key][3], best_sim
            self._misses += 1
            return False, None, (best_sim if best_sim >= 0.0 else None)

    def put(self, question: str, value: Any, namespace: str = "") -> None:
        """Store an answer under the embedded question (LRU-bounded)."""
        vec = _unit(self._embedder.embed([question])[0])
        key = self._key(question, namespace)
        now = self._clock()
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            elif len(self._entries) >= self._maxsize:
                self._entries.popitem(last=False)
                self._evictions += 1
            self._entries[key] = (now, namespace, vec, value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.last_similarity = None

    # -------------------------------------------------------------- stats
    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses + self._expirations
            return {
                "kind": "semantic",
                "threshold": self._threshold,
                "hits": self._hits,
                "misses": self._misses,
                "expirations": self._expirations,
                "evictions": self._evictions,
                "size": len(self._entries),
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _key(question: str, namespace: str) -> str:
        # \x00 never occurs in text, so (namespace, question) stays unambiguous.
        return f"{namespace}\x00{question}"

    def _drop_expired(self, now: float) -> None:
        """Lazy expiry sweep from the LRU front (oldest first)."""
        if not self._ttl:
            return
        while self._entries:
            stored_at = next(iter(self._entries.values()))[0]
            if now - stored_at <= self._ttl:
                break
            self._entries.popitem(last=False)
            self._expirations += 1
