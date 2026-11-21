"""Load documents from a directory into Document objects.

Supported formats: plain text / markdown (``.txt`` / ``.md``) and PDF
(``.pdf``, via the optional ``pypdf`` dependency — imported lazily so a
text-only install keeps working).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    doc_id: str      # stable id, e.g. "01_machine_learning_basics"
    title: str       # human-readable, derived from file name
    text: str
    source: str      # original file path (as string)


def _read_pdf(path: Path) -> str:
    """Extract per-page text from a PDF, pages joined by blank lines.

    ``pypdf`` is imported lazily: text-only users never pay for it. A PDF
    whose pages yield no text (scans, image-only) produces an empty string
    and the caller skips it with a visible warning — silently indexing an
    empty document would look like a retrieval bug later.
    """
    try:
        from pypdf import PdfReader
    except ImportError as err:
        raise ImportError(
            f"{path.name}: reading .pdf files needs pypdf — "
            "install it with `pip install pypdf`"
        ) from err

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(page for page in pages if page)


def load_documents(data_dir: str | Path) -> list[Document]:
    """Read every supported file in *data_dir* (non-recursive, sorted by name)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path).strip()
            if not text:
                print(
                    f"warning: {path.name} has no extractable text "
                    "(scanned/image-only PDF?) — skipped",
                    file=sys.stderr,
                )
                continue
        else:
            text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = path.stem
        title = doc_id.replace("_", " ").replace("-", " ")
        documents.append(Document(doc_id=doc_id, title=title, text=text, source=str(path)))

    if not documents:
        raise ValueError(f"No supported documents found in {data_dir} (.txt/.md/.pdf)")
    return documents
