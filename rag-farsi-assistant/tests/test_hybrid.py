"""Tests for RRF fusion and the HybridRetriever (dense / sparse / hybrid)."""
import pytest

from src.chunker import Chunk
from src.embedder import FakeHashEmbedder
from src.hybrid import HybridRetriever, rrf_fuse
from src.vector_store import VectorStore


def build_retriever(texts: list[str], mode: str = "hybrid", **kwargs) -> HybridRetriever:
    embedder = FakeHashEmbedder(dimension=128)
    store = VectorStore(dimension=128)
    vectors = embedder.embed(texts)
    chunks = [
        Chunk(chunk_id=f"d::{i}", doc_id="d", title="t", text=t, position=i)
        for i, t in enumerate(texts)
    ]
    store.add(vectors, chunks)
    return HybridRetriever(embedder, store, mode=mode, **kwargs)


class TestRRF:
    def test_known_fusion_scores(self):
        # doc "b" is ranked 1st by both rankers -> must win the fusion
        fused = rrf_fuse([["a", "b", "c"], ["b", "d"]], rrf_k=60)
        scores = dict(fused)
        assert fused[0][0] == "b"
        assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["a"] == pytest.approx(1 / 61)

    def test_consensus_beats_single_top_rank(self):
        # "x" is 1st for one ranker only, "y" is 1st AND 2nd -> "y" wins
        fused = rrf_fuse([["x", "y"], ["y", "z"]], rrf_k=60)
        assert fused[0][0] == "y"

    def test_top_k_and_invalid_k(self):
        fused = rrf_fuse([["a", "b", "c", "d"]], rrf_k=60, top_k=2)
        assert len(fused) == 2
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], rrf_k=0)


class TestHybridRetriever:
    def test_mode_validation(self):
        with pytest.raises(ValueError):
            build_retriever(["متن"], mode="magic")

    def test_sparse_mode_exact_lexical_match_wins(self):
        texts = [
            "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند",
            "خوشه‌بندی مشتریان نمونه‌ای از یادگیری بدون نظارت است",
            "فیلترینگ مشارکتی برای پیشنهاد فیلم استفاده می‌شود",
        ]
        retriever = build_retriever(texts, mode="sparse")
        results = retriever.retrieve("فیلترینگ مشارکتی فیلم")
        assert results[0].chunk.text == texts[2]
        # scores are min-max normalised into (0, 1]
        assert 0.0 < results[0].score <= 1.0

    def test_hybrid_surfaces_exact_term_match(self):
        # the rare term "پس‌انتشار" only appears in doc 0; hybrid must rank
        # it first even though the hash embedder knows nothing about meaning
        texts = [
            "پس‌انتشار خطا وزن‌های شبکه عصبی را به‌روزرسانی می‌کند",
            "شبکه کانولوشنی برای تحلیل تصویر طراحی شده است",
            "معماری ترنسفورمر بر مکانیزم توجه بنا شده است",
        ]
        retriever = build_retriever(texts, mode="hybrid")
        results = retriever.retrieve("پس‌انتشار در شبکه عصبی")
        assert results[0].chunk.text == texts[0]

    def test_hybrid_fuses_both_rankings(self):
        # chunk 0 shares vocabulary with the query, chunk 1 is dense-similar;
        # both must appear in the hybrid top-2
        texts = [
            "فیلترینگ مشارکتی امتیاز کاربران را مقایسه می‌کند",
            "فیلترینگ مشارکتی امتیاز کاربران را مقایسه می‌کند",
            "توابع فعال‌سازی غیرخطی به شبکه توانایی می‌دهند",
        ]
        retriever = build_retriever(texts, mode="hybrid")
        results = retriever.retrieve("فیلترینگ مشارکتی", top_k=2)
        assert len(results) == 2
        assert all(r.chunk.text.startswith("فیلترینگ") for r in results)

    def test_top_k_respected_each_mode(self):
        texts = [f"متن شماره {i} درباره یادگیری ماشین" for i in range(6)]
        for mode in ("dense", "sparse", "hybrid"):
            retriever = build_retriever(texts, mode=mode, top_k=3)
            assert len(retriever.retrieve("یادگیری ماشین")) == 3

    def test_empty_store_falls_back_gracefully(self):
        embedder = FakeHashEmbedder(dimension=128)
        store = VectorStore(dimension=128)
        retriever = HybridRetriever(embedder, store, mode="hybrid")
        assert retriever.retrieve("هر سؤالی") == []

    def test_store_swap_rebuilds_bm25(self):
        texts_a = ["فیلترینگ مشارکتی امتیازها را مقایسه می‌کند"]
        texts_b = ["انتقال یادگیری مدل از پیش آموزش‌دیده است"]
        retriever = build_retriever(texts_a, mode="sparse")
        assert retriever.retrieve("انتقال یادگیری") == []

        embedder = FakeHashEmbedder(dimension=128)
        new_store = VectorStore(dimension=128)
        new_store.add(
            embedder.embed(texts_b),
            [Chunk(chunk_id="n::0", doc_id="n", title="t", text=texts_b[0], position=0)],
        )
        retriever.store = new_store  # simulates pipeline.load_index()
        results = retriever.retrieve("انتقال یادگیری")
        assert results and results[0].chunk.text == texts_b[0]
