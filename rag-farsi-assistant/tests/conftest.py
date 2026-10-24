"""Shared pytest fixtures: tmp corpus, fake embedder, mock LLM, pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig  # noqa: E402
from src.embedder import FakeHashEmbedder  # noqa: E402
from src.llm.base import LLMProvider  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

DOC_ML = (
    "یادگیری ماشین شاخه‌ای از هوش مصنوعی است.\n\n"
    "یادگیری با نظارت از داده‌های برچسب‌دار استفاده می‌کند. دسته‌بندی ایمیل هرزنامه "
    "یک مثال رایج یادگیری با نظارت است. خوشه‌بندی یک روش یادگیری بدون نظارت است.\n\n"
    "بیش‌برازش یعنی مدل داده آموزش را حفظ کند. برای مقابله با بیش‌برازش از "
    "منظم‌سازی و اعتبارسنجی متقاطع استفاده می‌شود."
)

DOC_DL = (
    "شبکه عصبی از لایه‌های نورون ساخته می‌شود.\n\n"
    "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند. مکانیزم توجه یا Attention هسته "
    "معماری ترنسفورمر است. بهینه‌ساز Adam نرخ یادگیری را تطبیقی تنظیم می‌کند."
)


class MockLLM(LLMProvider):
    """Returns a deterministic answer string; records the last prompt."""

    name = "mock"

    def __init__(self):
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.calls: int = 0

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        return "پاسخ آزمایشی بر اساس منابع [1]."


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ml_basics.txt").write_text(DOC_ML, encoding="utf-8")
    (data_dir / "deep_learning.txt").write_text(DOC_DL, encoding="utf-8")
    return data_dir


@pytest.fixture()
def config(corpus_dir: Path, tmp_path: Path) -> RAGConfig:
    cfg = RAGConfig()
    cfg.data_dir = corpus_dir
    cfg.index_dir = tmp_path / "storage"
    cfg.embedding_backend = "fake"
    cfg.embedding_dimension = 128
    cfg.top_k = 2
    cfg.candidate_multiplier = 3
    return cfg


@pytest.fixture()
def fake_embedder(config: RAGConfig) -> FakeHashEmbedder:
    return FakeHashEmbedder(dimension=config.embedding_dimension)


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def pipeline(config: RAGConfig, fake_embedder: FakeHashEmbedder, mock_llm: MockLLM) -> RAGPipeline:
    return RAGPipeline(config, embedder=fake_embedder, llm=mock_llm)
