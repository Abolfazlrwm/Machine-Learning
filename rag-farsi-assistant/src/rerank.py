"""Optional re-ranking stage for the retrieval pipeline.

A cross-encoder reads (query, candidate) *jointly* and is therefore much
more precise than bi-encoder similarity — the classic trade-off is that it
cannot be pre-computed. The hook below keeps the pipeline agnostic:

* :class:`CrossEncoderReranker` wraps ``sentence_transformers.CrossEncoder``
  (e.g. ``BAAI/bge-reranker-v2-m3``, multilingual & Persian-capable). The
  heavy import happens lazily so the dependency stays optional.
* :class:`LexicalReranker` is a deterministic, dependency-free pointwise
  scorer — rarity-weighted query-term coverage over the candidate batch.
  It stands in for a neural cross-encoder in CI / offline runs and gives
  the re-rank stage something measurable to do without model downloads.
* :class:`NoopReranker` keeps the input order — useful in tests.

Wire one in via ``RAGConfig.reranker = "cross-encoder" | "lexical"`` (the
former needs ``pip install sentence-transformers``) or inject any
``BaseReranker`` into :class:`~src.hybrid.HybridRetriever` programmatically.
"""
from __future__ import annotations

import math

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Common interface: score (query, texts) pairs, higher = more relevant."""

    name: str = "base"

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text, aligned with the input order."""


class NoopReranker(BaseReranker):
    """Identity re-ranker: returns constant scores so the order is kept."""

    name = "noop"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


class LexicalReranker(BaseReranker):
    """Rarity-weighted lexical re-ranker — the offline cross-encoder stand-in.

    Scores each candidate by how much of the query it covers, weighting
    each query term by its rarity *within the candidate batch* (a poor
    man's IDF: ``w(t) = log(1 + N/df(t))``). Matching a distinctive term
    therefore beats matching a token every candidate shares. Deterministic,
    instant, no model download — which makes the re-rank stage measurable
    in CI and reproducible anywhere, and a drop-in swap for the neural
    cross-encoder at the same call site.

    Ties keep the incoming order (stable sort upstream), so candidates
    sharing no query term are left as the retriever ranked them.
    """

    name = "lexical"

    def score(self, query: str, texts: list[str]) -> list[float]:
        from .bm25 import tokenize  # imported lazily: keeps module import light

        q_tokens = set(tokenize(query))
        if not q_tokens or not texts:
            return [0.0] * len(texts)

        cand_tokens = [set(tokenize(t)) for t in texts]
        n = len(texts)
        # terms no candidate contains can never be matched: skip their weight
        weights = {
            t: math.log(1 + n / sum(1 for c in cand_tokens if t in c))
            for t in q_tokens
            if any(t in c for c in cand_tokens)
        }
        total = sum(weights.values())

        return [
            sum(weights[t] for t in q_tokens if t in c) / total if total else 0.0
            for c in cand_tokens
        ]


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder re-ranker backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        try:
            from sentence_transformers import CrossEncoder  # lazy heavy import
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "CrossEncoderReranker requires the 'sentence-transformers' package. "
                "Install it with: pip install sentence-transformers — or set "
                "RAG_RERANKER=none to run without re-ranking."
            ) from exc
        self.name = f"cross-encoder:{model_name}"
        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        predictions = self._model.predict(pairs)
        return [float(p) for p in predictions]


def create_reranker(backend: str, **kwargs) -> BaseReranker | None:
    """Factory: ``"none"`` -> None, ``"noop"`` -> NoopReranker,
    ``"lexical"`` -> LexicalReranker, ``"cross-encoder"`` -> CrossEncoderReranker."""
    backends = {
        "none": lambda **kw: None,
        "noop": lambda **kw: NoopReranker(),
        "lexical": lambda **kw: LexicalReranker(**kw),
        "cross-encoder": lambda **kw: CrossEncoderReranker(**kw),
    }
    if backend not in backends:
        raise ValueError(f"unknown reranker backend {backend!r}; choose from {sorted(backends)}")
    return backends[backend](**kwargs)
