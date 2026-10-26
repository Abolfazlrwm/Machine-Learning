"""Retrieval-only evaluation: Hit@k and MRR over golden QA pairs.

No LLM needed — measures whether the retriever surfaces the right document.

Examples:
    python evaluate.py                       # real embeddings
    python evaluate.py --backend fake        # dependency-free smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig  # noqa: E402
from src.embedder import create_embedder  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

EVAL_DIR = Path(__file__).parent / "evaluation"


def evaluate(pipeline: RAGPipeline, ks: tuple[int, ...] = (1, 2, 3)) -> dict:
    pairs = json.loads((EVAL_DIR / "qa_pairs.json").read_text(encoding="utf-8"))["pairs"]
    hits = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []

    for pair in pairs:
        results = pipeline.retrieve(pair["question"], top_k=max(ks))
        ranked_docs = [r.chunk.doc_id + ".txt" for r in results]
        expected = pair["expected_doc"]
        rr = 0.0
        for rank, doc in enumerate(ranked_docs, start=1):
            if doc == expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        for k in ks:
            if expected in ranked_docs[:k]:
                hits[k] += 1

    n = len(pairs)
    return {
        "n_questions": n,
        **{f"hit@{k}": round(hits[k] / n, 4) for k in ks},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--backend", choices=["sentence-transformers", "fake"], default="sentence-transformers")
    parser.add_argument("--rebuild", action="store_true", help="rebuild index before evaluating")
    args = parser.parse_args()

    config = RAGConfig()
    config.embedding_backend = args.backend
    embedder = create_embedder(
        config.embedding_backend, config.embedding_model, dimension=config.embedding_dimension
    )
    pipeline = RAGPipeline(config, embedder=embedder, llm=None)
    if args.rebuild or not pipeline.load_index():
        stats = pipeline.build_index()
        print(f"[index] rebuilt: {stats}")

    metrics = evaluate(pipeline)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
