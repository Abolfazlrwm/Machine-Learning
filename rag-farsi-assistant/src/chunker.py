"""Paragraph/sentence-aware text chunking with overlap.

Splits documents into overlapping chunks that respect natural boundaries
(paragraphs, then sentences) instead of cutting text mid-word. Works for
Persian and English alike because sentence enders include ؟ ! and .
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data_loader import Document

_SENTENCE_ENDERS = "。!?.؟…\n"
_SENTENCE_RE = re.compile(
    r"[^" + re.escape(_SENTENCE_ENDERS) + r"]+[" + re.escape(_SENTENCE_ENDERS) + r"]*",
    flags=re.UNICODE,
)


@dataclass
class Chunk:
    chunk_id: str       # "<doc_id>::<position>"
    doc_id: str
    title: str
    text: str
    position: int       # index of the chunk inside its document


def split_text(text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> list[str]:
    """Split *text* into chunks of at most ``chunk_size`` characters.

    The splitter works bottom-up: paragraphs -> sentences -> characters.
    Consecutive chunks share ``chunk_overlap`` characters so that answers
    spanning a boundary are still retrievable as a whole.
    """
    if chunk_size < 2 * chunk_overlap:
        raise ValueError("chunk_size must be at least 2x chunk_overlap")
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        parts = [s.strip() for s in _SENTENCE_RE.findall(para) if s.strip()]
        sentences.extend(parts)

    # Hard-split pathological sentences that alone exceed chunk_size.
    pieces: list[str] = []
    for sentence in sentences:
        while len(sentence) > chunk_size:
            pieces.append(sentence[:chunk_size])
            sentence = sentence[chunk_size - chunk_overlap:]
        pieces.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        extra = len(piece) + (1 if current else 0)  # +1 for the joining space
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            # Build overlap tail from the end of the previous chunk.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > chunk_overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current = tail.copy()
            current_len = sum(len(p) for p in current) + max(0, len(current) - 1)
            extra = len(piece) + (1 if current else 0)
        current.append(piece)
        current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 150
) -> list[Chunk]:
    """Chunk a list of documents, keeping provenance metadata on every chunk."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        for pos, text in enumerate(
            split_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{pos}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    position=pos,
                )
            )
    return all_chunks
