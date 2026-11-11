"""Embedding backends.

* SentenceTransformerEmbedder — real multilingual embeddings (default).
* FakeHashEmbedder           — deterministic hashing-trick vectors, zero heavy
  dependencies. Useful for CI, unit tests and quick smoke runs.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", flags=re.UNICODE)


class BaseEmbedder(ABC):
    """Common interface: embed a batch of texts into L2-normalised vectors."""

    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array with unit-norm rows."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps ``sentence-transformers`` models.

    E5-family models require ``query:`` / ``passage:`` prefixes; this class
    applies them automatically based on the model name.
    """

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._is_e5 = "e5" in model_name.lower()

    @property
    def dimension(self) -> int:
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self._model.get_sentence_embedding_dimension
        return int(get_dim())

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _normalize(vectors.astype("float32"))

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_e5:
            text = f"query: {text}"
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        return _normalize(vec.astype("float32").reshape(1, -1))[0]


class FakeHashEmbedder(BaseEmbedder):
    """Deterministic hashed bag-of-words embedder.

    No model download, no torch — good enough for tests and demos of the
    plumbing, *not* for semantic quality. Tokens (Persian + latin) are
    hashed into a fixed-size vector, weighted by term frequency.
    """

    def __init__(self, dimension: int = 384):
        self._dim = int(dimension)
        self.name = f"fake-hash-{self._dim}"

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_row(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
        return int.from_bytes(digest[:4], "little") % self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                vectors[i, self._hash_row(token)] += 1.0
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def create_embedder(backend: str, model_name: str, dimension: int = 384) -> BaseEmbedder:
    """Factory used by the pipeline/CLI/app."""
    backend = backend.strip().lower()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "fake":
        return FakeHashEmbedder(dimension=dimension)
    raise ValueError(
        f"Unknown embedding backend '{backend}'. "
        "Available: 'sentence-transformers', 'fake'."
    )
