"""Query retrieval with optional Maximal Marginal Relevance (MMR).

Plain top-k cosine search can return near-duplicate chunks. MMR re-ranks
candidates trading relevance against diversity:

    MMR = argmax lambda * sim(query, chunk) - (1-lambda) * max sim(chunk, selected)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunker import Chunk
from .embedder import BaseEmbedder
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        top_k: int = 4,
        mmr_lambda: float = 0.7,
        candidate_multiplier: int = 4,
    ):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        query_vec = self.embedder.embed_query(query)
        n_candidates = min(k * self.candidate_multiplier, len(self.store))
        candidates = self.store.search(query_vec, top_k=n_candidates)
        if not candidates:
            return []
        if self.mmr_lambda >= 0.999 or len(candidates) <= 1:
            return [RetrievalResult(c, s) for c, s in candidates[:k]]

        return self._mmr(query_vec, candidates, k)

    def _mmr(
        self, query_vec: np.ndarray, candidates: list[tuple[Chunk, float]], k: int
    ) -> list[RetrievalResult]:
        # Embed candidates once for the diversity term.
        cand_vectors = self.embedder.embed([c.text for c, _ in candidates])
        query_vec = query_vec.reshape(1, -1)
        relevance = np.array([score for _, score in candidates])
        selected: list[int] = []
        remaining = list(range(len(candidates)))

        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                mmr_scores = []
                for i in remaining:
                    redundancy = max(
                        float(cand_vectors[i] @ cand_vectors[j]) for j in selected
                    )
                    mmr_scores.append(
                        self.mmr_lambda * relevance[i] - (1 - self.mmr_lambda) * redundancy
                    )
                best = remaining[int(np.argmax(mmr_scores))]
            selected.append(best)
            remaining.remove(best)

        return [RetrievalResult(candidates[i][0], candidates[i][1]) for i in selected]
