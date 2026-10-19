"""Vector store with FAISS (preferred) and pure-NumPy fallback.

Vectors are L2-normalised by the embedder, so inner product == cosine
similarity. Chunk payloads are kept in a JSON sidecar and joined by row id.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunker import Chunk

_META_FILE = "meta.json"
_CHUNKS_FILE = "chunks.json"
_VECTORS_FILE = "vectors.npy"
_FAISS_FILE = "index.faiss"


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, self.dimension), dtype="float32")
        # faiss is optional; the NumPy path is a drop-in fallback
        try:
            import faiss  # noqa: F401

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.dimension)
        except ImportError:
            self._faiss = None
            self._index = None

    # ------------------------------------------------------------------ add
    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")
        if embeddings.size and embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != store dim {self.dimension}"
            )
        if self._faiss is not None:
            self._index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
        self._chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    # --------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if not len(self._chunks) or top_k <= 0:
            return []
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")
        if self._faiss is not None:
            scores, ids = self._index.search(query, min(top_k, len(self._chunks)))
            pairs = zip(ids[0].tolist(), scores[0].tolist())
        else:
            if not len(self._chunks):
                return []
            sims = self._vectors @ query[0]
            order = np.argsort(-sims)[: min(top_k, len(self._chunks))]
            pairs = [(int(i), float(sims[i])) for i in order]
        return [(self._chunks[i], float(score)) for i, score in pairs if i >= 0]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    # ------------------------------------------------------------ save/load
    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {"dimension": self.dimension, "count": len(self._chunks)}
        (index_dir / _META_FILE).write_text(json.dumps(meta), encoding="utf-8")
        (index_dir / _CHUNKS_FILE).write_text(
            json.dumps([c.__dict__ for c in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        if self._faiss is not None:
            self._faiss.write_index(self._index, str(index_dir / _FAISS_FILE))
        else:
            np.save(index_dir / _VECTORS_FILE, self._vectors)

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorStore":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / _META_FILE).read_text(encoding="utf-8"))
        store = cls(dimension=meta["dimension"])
        raw_chunks = json.loads((index_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw_chunks]
        if store._faiss is not None:
            index_path = index_dir / _FAISS_FILE
            if index_path.exists():
                store._index = store._faiss.read_index(str(index_path))
            else:  # index built on a machine without faiss -> load raw vectors
                vectors = np.load(index_dir / _VECTORS_FILE)
                store._index.add(vectors)
        else:
            store._vectors = np.load(index_dir / _VECTORS_FILE)
        store._chunks = chunks
        return store
