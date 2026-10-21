import numpy as np

from src.chunker import Chunk
from src.embedder import FakeHashEmbedder
from src.retriever import Retriever
from src.vector_store import VectorStore


def build_store(texts: list[str]) -> tuple[FakeHashEmbedder, VectorStore]:
    embedder = FakeHashEmbedder(dimension=128)
    store = VectorStore(dimension=128)
    vectors = embedder.embed(texts)
    chunks = [
        Chunk(chunk_id=f"d::{i}", doc_id="d", title="t", text=t, position=i)
        for i, t in enumerate(texts)
    ]
    store.add(vectors, chunks)
    return embedder, store


class TestRetriever:
    def test_relevant_document_ranks_first(self):
        texts = [
            "مکانیزم توجه هسته ترنسفورمر است",
            "خوشه‌بندی روش یادگیری بدون نظارت است",
            "فیلترینگ مشارکتی امتیاز کاربران را مقایسه می‌کند",
        ]
        embedder, store = build_store(texts)
        retriever = Retriever(embedder, store, top_k=2, mmr_lambda=1.0)
        results = retriever.retrieve("مکانیزم توجه در ترنسفورمر چگونه است؟")
        assert results[0].chunk.text == texts[0]

    def test_scores_descending_without_mmr(self):
        texts = ["الف ب ج", "الف ب", "الف"]
        embedder, store = build_store(texts)
        retriever = Retriever(embedder, store, top_k=3, mmr_lambda=1.0)
        results = retriever.retrieve("الف ب")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        texts = [f"متن شماره {i} درباره یادگیری ماشین" for i in range(6)]
        embedder, store = build_store(texts)
        retriever = Retriever(embedder, store, top_k=3)
        results = retriever.retrieve("یادگیری ماشین")
        assert len(results) == 3

    def test_mmr_reduces_redundancy(self):
        # two identical chunks + one distinct; MMR should still surface the
        # distinct one within top-2, pure relevance would return dupes first
        texts = [
            "شبکه عصبی وزن دارد وزن",
            "شبکه عصبی وزن دارد وزن",
            "سیستم توصیه‌گر فیلم پیشنهاد می‌دهد",
        ]
        embedder, store = build_store(texts)
        mmr = Retriever(embedder, store, top_k=2, mmr_lambda=0.3)
        plain = Retriever(embedder, store, top_k=2, mmr_lambda=1.0)

        mmr_texts = [r.chunk.text for r in mmr.retrieve("شبکه عصبی وزن")]
        plain_texts = [r.chunk.text for r in plain.retrieve("شبکه عصبی وزن")]
        assert len(set(mmr_texts)) == 2, "MMR must avoid duplicate chunks"
        assert plain_texts[0] == texts[0]

    def test_empty_store(self):
        embedder = FakeHashEmbedder(dimension=16)
        store = VectorStore(dimension=16)
        retriever = Retriever(embedder, store)
        assert retriever.retrieve("هر چیزی") == []
