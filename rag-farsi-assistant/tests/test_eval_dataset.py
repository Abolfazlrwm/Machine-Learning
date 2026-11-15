"""Golden dataset integrity + a retrieval regression gate.

The QA set is *data*, and broken data fails silently: a duplicated
question, a dangling ``expected_doc`` or a hollow gold answer degrades
every reported metric without breaking a single unit test. These checks
pin the dataset contract; the smoke test at the bottom runs the exact
``evaluate.py`` retrieval path so the committed baseline cannot drift.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
CORPUS = {p.name for p in (PROJECT / "data").glob("*.txt")}


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    return json.loads((PROJECT / "evaluation" / "qa_pairs.json").read_text("utf-8"))["pairs"]


def test_100_pairs_balanced_across_the_corpus(pairs):
    assert len(pairs) == 100
    dist = Counter(p["expected_doc"] for p in pairs)
    assert set(dist) == CORPUS, "every corpus document must carry questions"
    assert min(dist.values()) >= 14, "per-document coverage must stay balanced"


def test_questions_are_unique_and_nonempty(pairs):
    questions = [p["question"].strip() for p in pairs]
    assert all(questions)
    assert len(questions) == len(set(questions)), "paraphrases welcome, duplicates not"


def test_every_expected_doc_exists(pairs):
    bad = {p["expected_doc"] for p in pairs} - CORPUS
    assert not bad, f"dangling expected_doc: {bad}"


def test_gold_answers_are_substantive(pairs):
    for p in pairs:
        assert len(p["gold_answer"].strip()) >= 30, f"hollow gold answer: {p['question']}"


def test_pairs_carry_only_the_agreed_schema(pairs):
    assert all(set(p) == {"question", "expected_doc", "gold_answer"} for p in pairs)


def test_retrieval_smoke_hybrid_rerank_stays_strong():
    """Regression gate over the full golden set (deterministic, no downloads).

    Runs the same path as ``evaluate.py --backend fake`` (fresh 384-dim
    fake-hash index, hybrid retrieval + lexical re-ranker) and pins the
    committed baseline (hit@1 0.92 / hit@3 0.99 / MRR 0.9517) with a small
    margin so corpus or code changes surface loudly, not silently.
    """
    import evaluate as ev  # project root is on sys.path via conftest
    from src.config import RAGConfig
    from src.embedder import create_embedder
    from src.pipeline import RAGPipeline

    config = RAGConfig()
    config.embedding_backend = "fake"
    config.retrieval_mode = "hybrid"
    config.reranker = "lexical"
    embedder = create_embedder(
        "fake", config.embedding_model, dimension=config.embedding_dimension
    )
    pipeline = RAGPipeline(config, embedder=embedder, llm=None)
    pipeline.build_index()

    metrics = ev.evaluate(pipeline)
    assert metrics["hit@1"] >= 0.90, metrics
    assert metrics["hit@3"] >= 0.98, metrics
    assert metrics["mrr"] >= 0.93, metrics
