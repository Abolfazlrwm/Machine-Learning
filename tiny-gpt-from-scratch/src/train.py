"""Train TinyGPT on the Persian corpus and persist artifacts.

Usage (from tiny-gpt-from-scratch/):
    python -m src.train --steps 600 --lr 3e-3 --dim 96 --layers 2

Writes ``artifacts/tokenizer.json``, ``artifacts/params.npz``,
``artifacts/train_log.json`` (config + loss history + val loss) and
``artifacts/samples.txt`` — the committed artifacts the README reports.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import load_corpus, random_batch
from .model import TinyGPT
from .optimizer import Adam
from .tokenizer import CharTokenizer

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1379)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(args.seed)          # parameter init
    rng = np.random.default_rng(args.seed)  # batch + sampling randomness

    text = load_corpus()
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text), dtype=np.int64)
    split = int(0.95 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = TinyGPT(tok.vocab_size, dim=args.dim, n_layers=args.layers,
                    n_heads=args.heads, max_t=args.block)
    params = model.parameters()
    n_params = sum(p.data.size for p in params.values())
    opt = Adam(params, lr=args.lr)

    loss_history: list[dict] = []
    t0 = time.time()
    print(f"corpus: {len(data):,} chars | vocab: {tok.vocab_size} | "
          f"params: {n_params:,} | device: cpu (float64)")

    for step in range(1, args.steps + 1):
        batch = random_batch(train_data, args.block, args.batch, rng)
        loss = model.loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            val_batch = random_batch(val_data, args.block, args.batch, rng)
            val_loss = float(model.loss(val_batch).data)
            loss_history.append({"step": step, "train": float(loss.data),
                                 "val": val_loss})
            print(f"step {step:4d} | train {loss.data:.4f} | val {val_loss:.4f}")

    elapsed = time.time() - t0

    # ---- persist artifacts -------------------------------------------
    ARTIFACTS.mkdir(exist_ok=True)
    tok.to_json(ARTIFACTS / "tokenizer.json")
    np.savez(ARTIFACTS / "params.npz", **model.state_dict())

    config = {"dim": args.dim, "n_layers": args.layers, "n_heads": args.heads,
              "max_t": args.block, "vocab": tok.vocab_size,
              "n_params": int(n_params)}
    log = {"config": config, "train_args": vars(args),
           "corpus_chars": int(len(data)), "elapsed_sec": round(elapsed, 1),
           "loss_history": loss_history,
           "final_train_loss": loss_history[-1]["train"],
           "final_val_loss": loss_history[-1]["val"]}
    (ARTIFACTS / "train_log.json").write_text(json.dumps(log, indent=1),
                                              encoding="utf-8")

    prompt = tok.encode("یادگیری ماشین")
    ids = model.sample(prompt, n_new=280, temperature=0.8, top_k=12, rng=rng)
    sample_text = tok.decode(ids)
    (ARTIFACTS / "samples.txt").write_text(sample_text, encoding="utf-8")
    print(f"\ndone in {elapsed:.1f}s | final train {log['final_train_loss']:.4f} "
          f"| val {log['final_val_loss']:.4f}")
    print("sample >>>", sample_text[:160].replace("\n", " "))


if __name__ == "__main__":
    main()
