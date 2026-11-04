"""Tests for the from-scratch BM25 sparse retriever."""
from src.bm25 import BM25Okapi, normalize_fa, tokenize


class TestTokenizer:
    def test_arabic_to_persian_normalisation(self):
        # Arabic Yeh/Keh must match Persian forms
        assert normalize_fa("يلت كت") == "یلت کت"

    def test_zwnj_removed(self):
        assert normalize_fa("می\u200cشود") == "میشود"

    def test_stopwords_and_punctuation_dropped(self):
        tokens = tokenize("این، یک «آزمون» ساده است!")
        assert "این" not in tokens and "یک" not in tokens and "است" not in tokens
        assert "آزمون" in tokens and "ساده" in tokens

    def test_latin_terms_kept_and_lowered(self):
        assert set(tokenize("مدل BERT و RAG")) == {"مدل", "bert", "rag"}


class TestBM25:
    def make_index(self):
        docs = [
            "مکانیزم توجه هسته معماری ترنسفورمر است",
            "خوشه‌بندی روش یادگیری بدون نظارت است",
            "فیلترینگ مشارکتی امتیاز کاربران را مقایسه می‌کند",
        ]
        return BM25Okapi(docs)

    def test_relevant_document_ranks_first(self):
        bm25 = self.make_index()
        top = bm25.search("مکانیزم توجه در ترنسفورمر", top_k=3)
        assert top[0][0] == 0
        assert top[0][1] > 0

    def test_zero_score_documents_excluded(self):
        bm25 = self.make_index()
        hits = bm25.search("فیلترینگ مشارکتی", top_k=3)
        assert len(hits) == 1  # only the CF document shares vocabulary
        assert hits[0][0] == 2

    def test_top_k_respected(self):
        docs = [f"متن شماره {i} درباره یادگیری ماشین" for i in range(6)]
        bm25 = BM25Okapi(docs)
        assert len(bm25.search("یادگیری ماشین", top_k=3)) == 3

    def test_empty_query_and_empty_corpus(self):
        assert BM25Okapi([]).search("هرچی", top_k=2) == []
        bm25 = self.make_index()
        assert bm25.search("که را و", top_k=2) == []  # only stopwords

    def test_term_frequency_saturates(self):
        # tf=4 must score higher than tf=1 but far less than 4x
        bm25 = BM25Okapi(["توجه توجه توجه توجه", "توجه"])
        (s_high,) = [s for _, s in bm25.search("توجه", top_k=2) if _ == 0]
        (s_low,) = [s for _, s in bm25.search("توجه", top_k=2) if _ == 1]
        assert s_high > s_low
        assert s_high < 4 * s_low

    def test_invalid_constants_rejected(self):
        try:
            BM25Okapi(["متن"], k1=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("negative k1 must raise")
        try:
            BM25Okapi(["متن"], b=1.5)
        except ValueError:
            pass
        else:
            raise AssertionError("b outside [0,1] must raise")
