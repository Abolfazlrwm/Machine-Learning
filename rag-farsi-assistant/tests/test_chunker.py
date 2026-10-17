from src.chunker import chunk_documents, split_text
from src.data_loader import Document, load_documents

SAMPLE_TEXT = (
    "یادگیری ماشین شاخه‌ای از هوش مصنوعی است.\n\n"
    "یادگیری با نظارت از داده‌های برچسب‌دار استفاده می‌کند. دسته‌بندی ایمیل هرزنامه "
    "یک مثال رایج یادگیری با نظارت است."
)


class TestSplitText:
    def test_respects_chunk_size(self):
        text = ". ".join(f"جمله شماره {i} است" for i in range(120))
        chunks = split_text(text, chunk_size=300, chunk_overlap=60)
        assert chunks, "must produce at least one chunk"
        assert all(len(c) <= 300 for c in chunks)

    def test_all_sentences_preserved(self):
        sentence = "این یک جمله آزمایشی است"
        text = ". ".join([sentence] * 40)
        chunks = split_text(text, chunk_size=200, chunk_overlap=50)
        joined = " ".join(chunks)
        # overlap may repeat boundary sentences, but never lose any
        assert joined.count(sentence) >= 40

    def test_neighbouring_chunks_overlap(self):
        text = ". ".join(f"قانون شماره {i}" for i in range(50))
        chunks = split_text(text, chunk_size=180, chunk_overlap=50)
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = prev.split()[-2:]
            assert any(word in nxt.split() for word in tail), \
                "overlap must carry words from the previous chunk"

    def test_empty_text(self):
        assert split_text("") == []
        assert split_text("   \n\n  ") == []

    def test_invalid_overlap_raises(self):
        import pytest

        with pytest.raises(ValueError):
            split_text("abc", chunk_size=100, chunk_overlap=80)


class TestChunkDocuments:
    def test_provenance_metadata(self):
        doc = Document(doc_id="demo", title="demo title", text=SAMPLE_TEXT, source="demo.txt")
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=40)
        assert chunks
        assert all(c.doc_id == "demo" for c in chunks)
        assert all(c.chunk_id.startswith("demo::") for c in chunks)
        assert [c.position for c in chunks] == list(range(len(chunks)))
        assert all(c.title == "demo title" for c in chunks)

    def test_load_documents_sorted(self, tmp_path):
        (tmp_path / "b_second.txt").write_text("دوم", encoding="utf-8")
        (tmp_path / "a_first.txt").write_text("اول", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert [d.doc_id for d in docs] == ["a_first", "b_second"]

    def test_load_documents_skips_unsupported(self, tmp_path):
        (tmp_path / "keep.txt").write_text("متن", encoding="utf-8")
        (tmp_path / "ignore.csv").write_text("a,b", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert [d.doc_id for d in docs] == ["keep"]
