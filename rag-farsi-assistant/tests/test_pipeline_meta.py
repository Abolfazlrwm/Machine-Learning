"""Pipeline-level tests: meta blocks, cache behaviour and trace plumbing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAGConfig  # noqa: E402
from src.observability import Trace  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def test_answer_meta_with_cache_off(config, fake_embedder, mock_llm):
    config.cache_ttl = 0.0
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    result = pipe.answer("بیش‌برازش چیست؟")
    assert result.meta["cache"] == "off"
    assert {"retrieve", "generate"} <= set(result.meta["timings_ms"])
    assert result.meta["query"]["original"] == "بیش‌برازش چیست؟"


def test_answer_meta_reports_query_expansions(config, fake_embedder, mock_llm):
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    result = pipe.answer("overfitting چیست؟")
    assert "بیشبرازش" in result.meta["query"]["expansions"]
    assert result.meta["query"]["mixed_script"] is True


def test_cache_serves_repeat_questions_without_new_llm_calls(
    config, fake_embedder, mock_llm
):
    config.cache_ttl = 60.0
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    first = pipe.answer("بیش‌برازش چیست؟")
    second = pipe.answer("بیش‌برازش چیست؟")
    assert first.meta["cache"] == "miss"
    assert second.meta["cache"] == "hit"
    assert mock_llm.calls == 1                       # generation ran once
    assert second.answer == first.answer
    assert second.sources == first.sources


def test_cache_respects_retrieval_settings(config, fake_embedder, mock_llm):
    config.cache_ttl = 60.0
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست؟", top_k=2)
    pipe.answer("بیش‌برازش چیست؟", top_k=4)  # different k -> different key
    assert mock_llm.calls == 2


def test_cache_survives_persian_spelling_variants(config, fake_embedder, mock_llm):
    config.cache_ttl = 60.0
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست؟")      # with zero-width non-joiner
    hit = pipe.answer("بیشبرازش چیست؟")  # ZWNJ stripped by the fold
    assert hit.meta["cache"] == "hit"


def test_external_trace_collects_stage_timings(config, fake_embedder, mock_llm):
    trace = Trace()
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    pipe.answer("بیش‌برازش چیست؟", trace=trace)
    assert {"retrieve", "generate"} <= set(trace.stages)
    assert trace.notes["cache"] == "off"
    assert trace.notes["mode"] == "hybrid"


def test_no_index_answer_carries_meta(config, fake_embedder, mock_llm, monkeypatch):
    """Defensive branch: an index build that yields nothing still carries meta."""
    pipe = RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
    monkeypatch.setattr(pipe, "build_index", lambda: {"documents": 0, "chunks": 0})
    result = pipe.answer("بیش‌برازش چیست؟")
    assert result.meta["cache"] == "off"
    assert "build" in result.answer
