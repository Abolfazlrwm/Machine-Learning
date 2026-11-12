"""Tests for the optional re-ranking stage."""
import pytest

from src.hybrid import HybridRetriever
from src.rerank import BaseReranker, LexicalReranker, NoopReranker, create_reranker


class ReverseReranker(BaseReranker):
    """Test double: gives each text a score based on its position label."""

    name = "reverse"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float("d2" in t) for t in texts]


def test_noop_reranker_keeps_scores():
    assert NoopReranker().score("q", ["الف", "ب"]) == [0.0, 0.0]


class TestLexicalReranker:
    def test_distinctive_term_wins_over_shared_term(self):
        # both candidates share the generic tokens; only one has the rare one
        texts = ["یادگیری ماشین مدل‌ها را آموزش می‌دهد", "یادگیری ماشین کلاسترینگ"]
        scores = LexicalReranker().score("کلاسترینگ خوشه‌بندی", texts)
        assert scores[1] > scores[0]

    def test_full_coverage_beats_partial(self):
        texts = ["فقط شبکه", "شبکه عصبی مصنوعی"]
        scores = LexicalReranker().score("شبکه عصبی", texts)
        assert scores[1] > scores[0]

    def test_no_overlap_keeps_zero_scores(self):
        texts = ["متن اول", "متن دوم"]
        assert LexicalReranker().score("کاملاً نامرتبط", texts) == [0.0, 0.0]

    def test_empty_query_returns_zeros(self):
        assert LexicalReranker().score("   ", ["الف", "ب"]) == [0.0, 0.0]

    def test_scores_are_deterministic(self):
        texts = ["یادگیری عمیق شبکه‌ها", "خوشه‌بندی داده‌ها", "یادگیری عمیق"]
        first = LexicalReranker().score("یادگیری عمیق", texts)
        second = LexicalReranker().score("یادگیری عمیق", texts)
        assert first == second

    def test_factory_builds_lexical(self):
        assert isinstance(create_reranker("lexical"), LexicalReranker)


def test_reranker_reorders_results():
    from src.chunker import Chunk
    from src.embedder import FakeHashEmbedder
    from src.vector_store import VectorStore

    texts = ["متن اول درباره خوشه‌بندی", "متن دوم درباره خوشه‌بندی d2"]
    embedder = FakeHashEmbedder(dimension=128)
    store = VectorStore(dimension=128)
    store.add(
        embedder.embed(texts),
        [
            Chunk(chunk_id=f"d::{i}", doc_id="d", title="t", text=t, position=i)
            for i, t in enumerate(texts)
        ],
    )
    retriever = HybridRetriever(
        embedder, store, mode="dense", reranker=ReverseReranker()
    )
    results = retriever.retrieve("خوشه‌بندی", top_k=2)
    assert "d2" in results[0].chunk.text


def test_create_reranker_factory():
    assert create_reranker("none") is None
    assert isinstance(create_reranker("noop"), NoopReranker)
    with pytest.raises(ValueError):
        create_reranker("magic-model")


def test_cross_encoder_requires_dependency():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="sentence-transformers"):
            create_reranker("cross-encoder")
    else:
        pytest.skip("sentence-transformers installed; skip dependency guard test")
