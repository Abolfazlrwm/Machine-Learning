"""Load plain-text/markdown documents from a directory into Document objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class Document:
    doc_id: str      # stable id, e.g. "01_machine_learning_basics"
    title: str       # human-readable, derived from file name
    text: str
    source: str      # original file path (as string)


def load_documents(data_dir: str | Path) -> list[Document]:
    """Read every supported file in *data_dir* (non-recursive, sorted by name)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_id = path.stem
        title = doc_id.replace("_", " ").replace("-", " ")
        documents.append(Document(doc_id=doc_id, title=title, text=text, source=str(path)))

    if not documents:
        raise ValueError(f"No .txt/.md documents found in {data_dir}")
    return documents
