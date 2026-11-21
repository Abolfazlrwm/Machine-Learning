import pytest

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


def _minimal_pdf(text: str | None = "sample text") -> bytes:
    """Build a valid one-page PDF (Helvetica, Latin text) without any
    extra dependency — just enough for pypdf to round-trip *text*.
    ``text=None`` builds a content-less page (scanned/image-only case)."""
    has_content = text is not None
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
         if has_content else
         b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /Font << /F1 5 0 R >> >> >>"),
    ]
    if has_content:
        objects.append(b"<< /Length " + str(len(stream)).encode()
                       + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


class TestLoadDocumentsPdf:
    def test_pdf_text_extracted(self, tmp_path):
        (tmp_path / "report.pdf").write_bytes(_minimal_pdf("machine learning basics"))
        (docs) = load_documents(tmp_path)
        assert len(docs) == 1
        assert docs[0].doc_id == "report"
        assert "machine learning basics" in docs[0].text

    def test_pdf_and_txt_sorted_together(self, tmp_path):
        (tmp_path / "b_notes.pdf").write_bytes(_minimal_pdf("neural notes"))
        (tmp_path / "a_intro.txt").write_text("معرفی", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert [d.doc_id for d in docs] == ["a_intro", "b_notes"]

    def test_scanned_pdf_skipped_with_warning(self, tmp_path, capsys):
        # Valid structure, but the page has no content stream at all
        # (image-only scan) -> nothing extractable, skipped with a warning
        (tmp_path / "scan.pdf").write_bytes(_minimal_pdf(None))
        (tmp_path / "keep.txt").write_text("متن سالم", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert [d.doc_id for d in docs] == ["keep"]
        assert "scan.pdf" in capsys.readouterr().err

    def test_missing_pypdf_gives_install_hint(self, tmp_path, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        (tmp_path / "doc.pdf").write_bytes(_minimal_pdf("hello"))
        with pytest.raises(ImportError, match="pip install pypdf"):
            load_documents(tmp_path)
