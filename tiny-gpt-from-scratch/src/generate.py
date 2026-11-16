"""Sample from a trained TinyGPT checkpoint.

Usage (from tiny-gpt-from-scratch/):
    python -m src.generate --prompt "یادگیری ماشین" --chars 300 --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import TinyGPT
from .tokenizer import CharTokenizer

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def load_model(artifacts: Path = ARTIFACTS) -> tuple[TinyGPT, CharTokenizer]:
    tok = CharTokenizer.from_json(artifacts / "tokenizer.json")
    state = dict(np.load(artifacts / "params.npz"))
    cfg = state.pop("__config__", None)
    if cfg is None:  # config is not stored in the npz; use last-train defaults
        import json
        cfg = json.loads((artifacts / "train_log.json").read_text())["config"]
    model = TinyGPT(cfg["vocab"], dim=cfg["dim"], n_layers=cfg["n_layers"],
                    n_heads=cfg["n_heads"], max_t=cfg["max_t"])
    model.load_state_dict({k: np.asarray(v) for k, v in state.items()})
    return model, tok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="یادگیری ماشین")
    parser.add_argument("--chars", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model, tok = load_model()
    rng = np.random.default_rng(args.seed)
    prompt_ids = tok.encode(args.prompt)
    ids = model.sample(prompt_ids, n_new=args.chars, temperature=args.temperature,
                       top_k=args.top_k, rng=rng)
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
