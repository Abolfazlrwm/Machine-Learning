"""Character-level tokenizer, Persian-aware.

Vocabulary = every distinct character in the corpus (Persian letters,
ZWNJ, punctuation, digits, ASCII). Persian diacritic-free text is a
natural fit for char-level modelling: the vocabulary stays ~80 tokens
and every word remains fully compositional.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, vocab: str):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls("".join(sorted(set(text))))

    @classmethod
    def from_json(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["vocab"])

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}, ensure_ascii=False),
                              encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
