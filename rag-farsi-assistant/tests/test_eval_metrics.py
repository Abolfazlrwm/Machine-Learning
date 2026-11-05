"""Tests for the RAGAS-lite evaluation metrics — hand-computed fixtures."""
from evaluation.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    evaluate_generation,
    faithfulness,
    split_sentences,
    token_f1,
)


class TestTokenF1:
    def test_identical_texts_score_one(self):
        assert token_f1("شبکه عصبی چندلایه", "شبکه عصبی چندلایه") == 1.0

    def test_disjoint_texts_score_zero(self):
        assert token_f1("شبکه عصبی", "فیلترینگ مشارکتی") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = token_f1("شبکه عصبی عمیق", "شبکه کانولوشنی")
        assert 0.0 < score < 1.0


class TestContextPrecision:
    def test_perfect_ranking_scores_one(self):
        gold = "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند"
        assert context_precision([gold, "متن بی‌ربط"], gold) == 1.0

    def test_relevant_chunk_ranked_last_scores_less(self):
        gold = "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند"
        score = context_precision(["متن بی‌ربط", gold], gold)
        assert 0.0 < score < 1.0

    def test_no_relevant_chunks_scores_zero(self):
        assert context_precision(["الف ب", "ج د"], "شبکه عصبی") == 0.0

    def test_empty_retrieval_scores_zero(self):
        assert context_precision([], "شبکه عصبی") == 0.0


class TestContextRecall:
    def test_full_coverage_is_one(self):
        gold = "ترنسفورمر مکانیزم توجه دارد"
        assert context_recall([gold], gold) == 1.0

    def test_partial_coverage_fraction(self):
        # gold has tokens {ترنسفورمر, مکانیزم, توجه, دارد}; context covers 2
        score = context_recall(["مکانیزم توجه هسته معماری است"], "ترنسفورمر مکانیزم توجه دارد")
        assert score == 0.5


class TestFaithfulness:
    def test_grounded_answer_scores_one(self):
        contexts = ["پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند و Adam بهینه‌ساز است"]
        answer = "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند. Adam بهینه‌ساز است."
        assert faithfulness(answer, contexts) == 1.0

    def test_half_grounded_answer_scores_half(self):
        contexts = ["پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند"]
        answer = "پس‌انتشار خطا وزن‌ها را به‌روزرسانی می‌کند. فیلترینگ مشارکتی فیلم پیشنهاد می‌دهد."
        assert faithfulness(answer, contexts) == 0.5


class TestAnswerRelevancy:
    def test_same_question_scores_one_without_embedder(self):
        q = "بیش‌برازش چیست و چطور مقابله می‌شود؟"
        assert answer_relevancy(q, q) == 1.0

    def test_unrelated_answer_scores_lower(self):
        q = "بیش‌برازش چیست و چطور مقابله می‌شود؟"
        assert answer_relevancy(q, "فیلترینگ مشارکتی فیلم پیشنهاد می‌دهد") < 1.0


def test_split_sentences_persian_enders():
    sentences = split_sentences("اول اینجا است. دوم اینجا است؟ سوم اینجا\nچهارم")
    assert len(sentences) == 4


def test_evaluate_generation_on_pipeline(pipeline):
    # the conftest `pipeline` fixture uses a fake embedder + mock LLM
    pairs = [
        {
            "question": "بیش‌برازش چیست؟",
            "expected_doc": "ml_basics.txt",
            "gold_answer": "بیش‌برازش یعنی مدل داده آموزش را حفظ کند.",
        }
    ]
    report = evaluate_generation(pairs, pipeline, top_k=2)
    assert report["n_questions"] == 1
    assert 0.0 <= report["context_recall"] <= 1.0
    assert report["faithfulness"] >= 0.0
    assert report["rows"][0]["question"] == "بیش‌برازش چیست؟"
