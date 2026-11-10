"""Hybrid retrieval: dense vectors + BM25 fused with Reciprocal Rank Fusion.

Dense embeddings capture semantic similarity, BM25 captures exact lexical
overlap. Their score scales are incomparable (cosine ∈ [-1, 1] vs unbounded
BM25), so instead of mixing raw scores we fuse *rankings*:

    RRF(d) = Σ_rankers 1 / (rrf_k + rank_ranker(d))

with the classic constant ``rrf_k = 60`` (Cormack et al., 2009). A document
ranked 1st by both systems scores ~2 × 1/61, while a document ranked 1st by
only one scores ~1/61 — consensus wins.

Three modes are supported:

* ``dense``  — embeddings only, MMR re-ranked (previous behaviour);
* ``sparse`` — BM25 only, scores min-max normalised for display;
* ``hybrid`` — both candidate lists fused with RRF (default).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .bm25 import BM25Okapi
from .chunker import Chunk
from .embedder import BaseEmbedder
from .observability import Trace
from .retriever import Retriever, RetrievalResult
from .rerank import BaseReranker
from .vector_store import VectorStore

DEFAULT_RRF_K = 60
VALID_MODES = ("dense", "sparse", "hybrid")


def rrf_fuse(
    rankings: list[list[str]],
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several rankings of ``chunk_id`` strings into one.

    Each ranking is an ordered list of chunk ids (best first). Returns
    ``[(chunk_id, rrf_score)]`` sorted by fused score, descending.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered[:top_k] if top_k else ordered


def _min_max(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Min-max normalise scores of an ``(index, score)`` list into (0, 1]."""
    if not pairs:
        return []
    raw = np.array([s for _, s in pairs], dtype="float64")
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-12:
        return [(i, 1.0) for i, _ in pairs]
    scaled = (raw - lo) / (hi - lo)
    return [(i, float(s)) for (i, _), s in zip(pairs, scaled)]


class HybridRetriever:
    """Retriever with switchable ``dense`` / ``sparse`` / ``hybrid`` modes.

    Dense mode delegates to :class:`~src.retriever.Retriever` (keeps MMR
    semantics). Sparse and hybrid modes build a BM25 index lazily from the
    chunks currently in the vector store; the index is rebuilt automatically
    whenever the store is replaced or grows.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        top_k: int = 4,
        mode: str = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = 4,
        mmr_lambda: float = 0.7,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        reranker: BaseReranker | None = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.embedder = embedder
        self._store = store
        self.top_k = top_k
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.reranker = reranker
        self._dense = Retriever(
            embedder=embedder,
            store=store,
            top_k=top_k,
            mmr_lambda=mmr_lambda,
            candidate_multiplier=candidate_multiplier,
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_len = -1

    # ------------------------------------------------------------ store swap
    @property
    def store(self) -> VectorStore:
        return self._store

    @store.setter
    def store(self, new_store: VectorStore) -> None:
        """Keep the dense path and the BM25 index in sync on index reloads."""
        self._store = new_store
        self._dense.store = new_store
        self._bm25_len = -1  # force BM25 rebuild from the new store

    # ------------------------------------------------------------------ bm25
    def _ensure_bm25(self) -> BM25Okapi | None:
        if self._bm25 is None or self._bm25_len != len(self._store):
            texts = [c.text for c in self._store.all_chunks()]
            self._bm25 = BM25Okapi(texts, k1=self.bm25_k1, b=self.bm25_b) if texts else None
            self._bm25_len = len(self.store)
        return self._bm25

    # -------------------------------------------------------------- retrieval
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        trace: Trace | None = None,
        embed_text: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates for ``query``.

        ``trace`` (optional) receives cumulative per-stage timings
        (``dense`` / ``bm25`` / ``rrf`` / ``rerank``) and candidate counts.

        ``embed_text`` decouples the two sub-queries of hybrid retrieval:
        BM25 ranks against ``query`` (which may carry cross-language
        expansion terms added by :mod:`src.query`) while the dense
        sub-search embeds ``embed_text`` — the visitor's original wording —
        so expansion never perturbs the evaluated E5 behaviour.
        """
        k = top_k or self.top_k
        tr = trace if trace is not None else Trace()
        tr.note("mode", self.mode)
        embed_query_text = embed_text or query

        if self.mode == "dense":
            with tr.span("dense"):
                return self._apply_reranker(query, self._dense.retrieve(query, k), tr)

        with tr.span("bm25"):
            bm25 = self._ensure_bm25()
        if bm25 is None:  # empty store -> graceful fallback to dense path
            with tr.span("dense"):
                return self._apply_reranker(
                    embed_query_text, self._dense.retrieve(embed_query_text, k), tr
                )

        n_candidates = min(k * self.candidate_multiplier, len(self._store))
        chunk_list = self._store.all_chunks()

        with tr.span("bm25"):
            sparse_hits = _min_max(bm25.search(query, n_candidates))
        tr.note("bm25_candidates", len(sparse_hits))

        if self.mode == "sparse":
            results = [
                RetrievalResult(chunk_list[i], score) for i, score in sparse_hits[:k]
            ]
            return self._apply_reranker(query, results, tr)

        # ---- hybrid: rank-based fusion of the two candidate lists ----
        with tr.span("dense"):
            query_vec = self.embedder.embed_query(embed_query_text)
            dense_pairs = self._store.search(query_vec, top_k=n_candidates)
        tr.note("dense_candidates", len(dense_pairs))
        dense_by_id = {c.chunk_id: (c, s) for c, s in dense_pairs}

        rankings = [
            [c.chunk_id for c, _ in dense_pairs],
            [chunk_list[i].chunk_id for i, _ in sparse_hits],
        ]
        with tr.span("rrf"):
            fused = rrf_fuse(rankings, rrf_k=self.rrf_k, top_k=k)

        results: list[RetrievalResult] = []
        for chunk_id, _rrf_score in fused:
            if chunk_id in dense_by_id:
                chunk, score = dense_by_id[chunk_id]
            else:
                idx = next(i for i, _ in sparse_hits if chunk_list[i].chunk_id == chunk_id)
                chunk, score = chunk_list[idx], dict(sparse_hits)[idx]
            results.append(RetrievalResult(chunk, score))
        return self._apply_reranker(query, results, tr)

    # ---------------------------------------------------------------- helpers
    def _apply_reranker(
        self,
        query: str,
        results: list[RetrievalResult],
        trace: Trace | None = None,
    ) -> list[RetrievalResult]:
        """Optional cross-encoder stage: re-order the final candidate list."""
        if self.reranker is None or len(results) <= 1:
            return results
        texts = [r.chunk.text for r in results]
        tr = trace if trace is not None else Trace()
        with tr.span("rerank"):
            scores = self.reranker.score(query, texts)
        order = sorted(range(len(results)), key=lambda i: -scores[i])
        return [results[i] for i in order]
