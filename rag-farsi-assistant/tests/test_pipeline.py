from src.config import RAGConfig
from src.llm.factory import create_provider
from src.pipeline import RAGPipeline


class TestPipeline:
    def test_build_index_stats(self, pipeline: RAGPipeline):
        stats = pipeline.build_index()
        assert stats["documents"] == 2
        assert stats["chunks"] >= 2

    def test_answer_uses_mock_llm_and_cites(self, pipeline: RAGPipeline, mock_llm):
        pipeline.build_index()
        result = pipeline.answer("بیش‌برازش چیست؟")
        assert result.answer == "پاسخ آزمایشی بر اساس منابع [1]."
        assert mock_llm.calls == 1
        assert result.sources, "must attach retrieved sources"
        assert all(0.0 <= s.score <= 1.0 for s in result.sources)
        assert len(result.sources) <= pipeline.config.top_k

    def test_prompt_contains_context_and_sources(self, pipeline: RAGPipeline, mock_llm):
        pipeline.build_index()
        pipeline.answer("پس‌انتشار خطا چگونه کار می‌کند؟")
        user_prompt = mock_llm.last_user
        assert "پس‌انتشار" in user_prompt
        assert "[1]" in user_prompt          # numbered context block
        assert "پرسش کاربر" in user_prompt   # question section header

    def test_system_prompt_is_strict(self, pipeline: RAGPipeline, mock_llm):
        pipeline.build_index()
        pipeline.answer("تست")
        system = mock_llm.last_system
        assert "فقط" in system           # only from context
        assert "نمی‌توانم" in system     # admit when unsure

    def test_extractive_fallback_without_llm(self, config: RAGConfig, fake_embedder):
        pipeline = RAGPipeline(config, embedder=fake_embedder, llm=None)
        pipeline.build_index()
        result = pipeline.answer("بیش‌برازش چیست؟")
        assert "[1]" in result.answer    # chunk texts numbered inline

    def test_save_load_index_equivalence(self, pipeline: RAGPipeline, config: RAGConfig,
                                         fake_embedder):
        pipeline.build_index()
        fresh = RAGPipeline(config, embedder=fake_embedder, llm=None)
        assert fresh.load_index()
        q = "مکانیزم توجه چیست؟"
        r1 = pipeline.retrieve(q, top_k=1)[0].chunk.chunk_id
        r2 = fresh.retrieve(q, top_k=1)[0].chunk.chunk_id
        assert r1 == r2

    def test_stats_after_build(self, pipeline: RAGPipeline):
        pipeline.build_index()
        stats = pipeline.stats()
        assert stats["documents"] == 2
        assert stats["chunks"] > 0
        assert "fake-hash" in stats["backend"]


def test_factory_unknown_provider(config: RAGConfig):
    import pytest

    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider(config, provider="claude")
