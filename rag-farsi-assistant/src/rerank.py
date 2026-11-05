"""Optional re-ranking stage for the retrieval pipeline.

A cross-encoder reads (query, candidate) *jointly* and is therefore much
more precise than bi-encoder similarity — the classic trade-off is that it
cannot be pre-computed. The hook below keeps the pipeline agnostic:

* :class:`CrossEncoderReranker` wraps ``sentence_transformers.CrossEncoder``
  (e.g. ``BAAI/bge-reranker-v2-m3``, multilingual & Persian-capable). The
  heavy import happens lazily so the dependency stays optional.
* :class:`NoopReranker` keeps the input order — useful in tests.

Wire one in via ``RAGConfig.reranker = "cross-encoder"`` (needs
``pip install sentence-transformers``) or inject any ``BaseReranker`` into
:class:`~src.hybrid.HybridRetriever` programmatically.
"""
from __future__ import annotations

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
    ``"cross-encoder"`` -> CrossEncoderReranker."""
    backends = {
        "none": lambda **kw: None,
        "noop": lambda **kw: NoopReranker(),
        "cross-encoder": lambda **kw: CrossEncoderReranker(**kw),
    }
    if backend not in backends:
        raise ValueError(f"unknown reranker backend {backend!r}; choose from {sorted(backends)}")
    return backends[backend](**kwargs)
