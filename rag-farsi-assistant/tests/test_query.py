"""Tests for the bilingual query understanding layer (src/query.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bm25 import BM25Okapi  # noqa: E402
from src.query import QueryPlan, analyse_query, fold, script_ratios  # noqa: E402

from tests.conftest import DOC_ML  # noqa: E402


# ------------------------------------------------------------- folding
def test_fold_maps_arabic_to_persian_chars():
    assert fold("بیش\u200cبرازش") == fold("بیشبرازش")
    assert fold("رایج") == fold("را\u064A\u062C")  # Arabic Yeh -> Persian


def test_fold_strips_zwnj_and_casefolds_latin():
    assert fold("Over\u200cFitting") == "overfitting"
    assert fold("MIT License") == "mit license"


def test_script_ratios_pure_and_empty():
    assert script_ratios("فارسی خالص")[0] == 1.0
    assert script_ratios("only latin")[1] == 1.0
    assert script_ratios("123 !؟") == (0.0, 0.0)


# ------------------------------------------------------------ analysis
def test_persian_query_gains_english_counterpart():
    plan = analyse_query("بیش‌برازش چیست و چطور جلوی آن را گرفت؟")
    assert "overfitting" in plan.expansions
    assert "overfitting" in plan.expanded
    assert plan.fa_ratio == pytest.approx(1.0)
    assert not plan.is_mixed


def test_english_query_gains_persian_counterpart():
    plan = analyse_query("overfitting چیست؟")
    assert plan.expansions == ["بیشبرازش"]  # folded: ZWNJ already stripped
    assert plan.is_mixed                    # Latin + Persian letters


def test_mixed_query_with_both_sides_needs_no_expansion():
    plan = analyse_query("بیش‌برازش (overfitting) یعنی چه؟")
    assert plan.expansions == []
    assert plan.expanded == plan.normalised


def test_unknown_vocabulary_stays_untouched():
    plan = analyse_query("قانون پارکینگ شهرداری چیست؟")
    assert plan.expansions == []


def test_plan_is_a_pure_function_of_the_input():
    a, b = analyse_query("data drift چیست؟"), analyse_query("data drift چیست؟")
    assert a == b
    assert isinstance(a, QueryPlan)


def test_expansions_are_capped():
    plan = analyse_query("attention transformer llm embedding overfitting spam")
    assert len(plan.expansions) <= 6


def test_normalised_form_strips_zwnj_for_comparison():
    plan = analyse_query("می‌شود بیش‌برازش را سنجید؟")
    assert "میشود" in plan.normalised
    assert "بیشبرازش" in plan.normalised


# --------------------------------------------- retrieval integration
def test_expansion_closes_the_lexical_gap_for_bm25():
    paragraphs = DOC_ML.split("\n\n")
    texts = [paragraphs[0], paragraphs[2]]  # [intro, overfitting paragraph]
    bm25 = BM25Okapi(texts)

    plain_hits = bm25.search("overfitting چیست؟", top_k=5)
    assert plain_hits == []  # zero lexical overlap without expansion

    expanded = analyse_query("overfitting چیست؟").expanded
    hits = bm25.search(expanded, top_k=5)
    assert hits and hits[0][0] == 1  # the overfitting paragraph wins


def test_dense_subquery_keeps_the_original_wording(pipeline):
    """Expansion must not perturb the dense side (eval numbers stay valid)."""
    embed_queries: list[str] = []
    original_embed = pipeline.embedder.embed_query

    def spy(text: str):
        embed_queries.append(text)
        return original_embed(text)

    pipeline.embedder.embed_query = spy  # type: ignore[method-assign]
    pipeline.answer("بیش‌برازش چیست؟")
    # The hybrid dense sub-search embedded the *original* question, not the
    # expanded string (which only feeds BM25).
    assert embed_queries and embed_queries[-1] == "بیش‌برازش چیست؟"
