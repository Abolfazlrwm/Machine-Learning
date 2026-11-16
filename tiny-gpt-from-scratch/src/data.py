"""Corpus loading and batch sampling for char-level training.

The default corpus is the same six Persian documents the RAG assistant
retrieves over (copied into ``corpus/`` so this project stays
standalone). Batches are (B, T+1) int matrices: ``inputs = x[:, :-1]``,
``targets = x[:, 1:]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> str:
    """Concatenate all ``*.txt`` files in the corpus directory."""
    files = sorted(Path(corpus_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no *.txt corpus files under {corpus_dir}")
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


def random_batch(data: np.ndarray, block_size: int, batch_size: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Sample a (B, T+1) batch of contiguous token windows."""
    starts = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    return np.stack([data[s:s + block_size + 1] for s in starts])
