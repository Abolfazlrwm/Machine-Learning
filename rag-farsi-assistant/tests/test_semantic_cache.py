"""Semantic cache: paraphrase-aware answer reuse (unit + pipeline tests)."""
from __future__ import annotations

import pytest

from src.embedder import FakeHashEmbedder
from src.semantic_cache import SemanticCache

from .conftest import MockLLM


class FakeClock:
    """Injectable monotonic clock the tests can advance."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def embedder() -> FakeHashEmbedder:
    return FakeHashEmbedder(dimension=64)


# --------------------------------------------------------------- unit tests
def test_exact_repeat_hits_with_similarity_one(embedder, clock):
    cache = SemanticCache(embedder, threshold=1.0, clock=clock)
    cache.put("بیش‌برازش چیست؟", ("answer", ["src"]))
    hit, value, sim = cache.lookup("بیش‌برازش چیست؟")
    assert hit is True
    assert value == ("answer", ["src"])
    assert sim == 1.0  # byte-equal vectors never fall to float rounding


def test_same_wording_in_different_order_hits(embedder, clock):
    # The fake embedder is a bag of words: identical token sets embed to the
    # identical vector, so word order / punctuation cannot dodge the cache.
    cache = SemanticCache(embedder, threshold=0.99, clock=clock)
    cache.put("بیش‌برازش چیست", "a")
    hit, value, sim = cache.lookup("چیست بیش‌برازش")
    assert hit and value == "a" and sim == 1.0


def test_paraphrase_hits_when_above_threshold(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.5, clock=clock)
    cache.put("بیش‌برازش چیست", "answer-1")
    # Extra shared token keeps the vectors close but not identical.
    hit, value, sim = cache.lookup("بیش‌برازش یعنی چیست")
    assert hit and value == "answer-1"
    assert 0.0 < sim < 1.0


def test_unrelated_question_misses(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, clock=clock)
    cache.put("بیش‌برازش چیست", "answer-1")
    hit, value, sim = cache.lookup("شبکه عصبی چگونه کار می‌کند")
    assert hit is False and value is None
    assert sim is not None and sim < 0.92  # near-miss similarity is reported


def test_threshold_gates_the_same_pair(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.3, clock=clock)
    cache.put("بیش‌برازش چیست", "answer-1")
    hit, _, sim = cache.lookup("بیش‌برازش یعنی چیست")
    assert hit and sim is not None
    strict = SemanticCache(embedder, threshold=min(sim + 0.01, 1.0), clock=clock)
    strict.put("بیش‌برازش چیست", "answer-1")
    hit2, _, _ = strict.lookup("بیش‌برازش یعنی چیست")
    assert hit2 is False


def test_ttl_expiry_with_injectable_clock(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, ttl=100.0, clock=clock)
    cache.put("بیش‌برازش چیست", "answer-1")
    clock.now += 101.0
    hit, _, _ = cache.lookup("بیش‌برازش چیست")
    assert hit is False
    assert cache.stats["expirations"] == 1
    assert cache.stats["size"] == 0


def test_lru_eviction_and_stats(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, maxsize=2, clock=clock)
    cache.put("q1", "a1")
    cache.put("q2", "a2")
    cache.put("q3", "a3")  # evicts q1 (least recently used)
    assert cache.stats["evictions"] == 1
    assert cache.stats["size"] == 2
    hit, value, _ = cache.lookup("q1")
    assert hit is False
    hit2, value2, _ = cache.lookup("q3")
    assert hit2 and value2 == "a3"
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1
    assert cache.stats["hit_rate"] == 0.5


def test_namespace_isolates_settings(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, clock=clock)
    cache.put("بیش‌برازش چیست", "top4-answer", namespace="4|hybrid")
    hit, _, _ = cache.lookup("بیش‌برازش چیست", namespace="8|hybrid")
    assert hit is False  # same question, different settings -> different cache
    hit2, value, _ = cache.lookup("بیش‌برازش چیست", namespace="4|hybrid")
    assert hit2 and value == "top4-answer"


def test_zero_ttl_never_expires(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, ttl=0.0, clock=clock)
    cache.put("بیش‌برازش چیست", "answer-1")
    clock.now += 10**9
    hit, value, _ = cache.lookup("بیش‌برازش چیست")
    assert hit and value == "answer-1"


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold_raises(embedder, threshold):
    with pytest.raises(ValueError):
        SemanticCache(embedder, threshold=threshold)


def test_invalid_bounds_raise(embedder):
    with pytest.raises(ValueError):
        SemanticCache(embedder, maxsize=0)
    with pytest.raises(ValueError):
        SemanticCache(embedder, ttl=-1.0)


def test_empty_cache_miss_reports_no_similarity(embedder, clock):
    cache = SemanticCache(embedder, threshold=0.92, clock=clock)
    hit, value, sim = cache.lookup("هر چیزی")
    assert hit is False and value is None and sim is None


# ---------------------------------------------------------- pipeline wiring
def test_pipeline_semantic_hit_on_repeated_question(config, fake_embedder, mock_llm):
    config.cache_ttl = 600.0
    config.semantic_cache = True
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    first = pipe.answer("بیش‌برازش چیست؟")
    assert first.meta["cache"] == "miss"
    second = pipe.answer("بیش‌برازش چیست؟")
    assert second.meta["cache"] == "hit"
    assert second.meta["cache_similarity"] == 1.0
    assert second.answer == first.answer
    assert mock_llm.calls == 1  # generation ran exactly once


def test_pipeline_semantic_hit_on_word_order_swap(config, fake_embedder, mock_llm):
    config.cache_ttl = 600.0
    config.semantic_cache = True
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست")
    second = pipe.answer("چیست بیش‌برازش")
    assert second.meta["cache"] == "hit"


def test_pipeline_semantic_miss_on_distinct_question(config, fake_embedder, mock_llm):
    config.cache_ttl = 600.0
    config.semantic_cache = True
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست")
    second = pipe.answer("شبکه عصبی چیست")
    assert second.meta["cache"] == "miss"
    assert mock_llm.calls == 2


def test_pipeline_cache_off_by_default_keeps_determinism(config, fake_embedder, mock_llm):
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    first = pipe.answer("بیش‌برازش چیست")
    second = pipe.answer("بیش‌برازش چیست")
    assert first.meta["cache"] == "off"
    assert second.meta["cache"] == "off"
    assert "cache" not in pipe.stats()


def test_pipeline_stats_expose_cache_counters(config, fake_embedder, mock_llm):
    config.cache_ttl = 600.0
    config.semantic_cache = True
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست")
    pipe.answer("بیش‌برازش چیست")
    stats = pipe.stats()["cache"]
    assert stats["kind"] == "semantic"
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["threshold"] == config.semantic_cache_threshold


def test_env_knobs_reach_the_config(monkeypatch):
    from src.config import RAGConfig

    monkeypatch.setenv("RAG_SEMANTIC_CACHE", "1")
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.87")
    cfg = RAGConfig.from_env()
    assert cfg.semantic_cache is True
    assert cfg.semantic_cache_threshold == pytest.approx(0.87)


def test_exact_ttl_cache_still_default_when_semantic_disabled(
    config, fake_embedder, mock_llm
):
    config.cache_ttl = 600.0
    config.semantic_cache = False
    from src.pipeline import RAGPipeline

    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    from src.cache import TTLCache

    assert isinstance(pipe.cache, TTLCache)
