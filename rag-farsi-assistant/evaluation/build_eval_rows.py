"""Per-question evaluation rows — the raw material for error analysis.

``evaluate.py`` prints aggregates; debugging needs the *rows*. This tool
re-runs the exact evaluation path (fake backend, fresh 384-dim index,
top-3 per mode) and writes ``evaluation/results/eval_rows.json``:

* ``retrieval_ranks``   — rank of the expected document per question, per
  retrieval mode (dense / sparse / hybrid / hybrid+rerank) plus the top-1
  document, so failure families (RRF dilution, cross-document overlap,
  lexical gaps) can be computed instead of guessed;
* ``generation_rows``   — the RAGAS-style per-question rows from
  ``evaluation.metrics.evaluate_generation`` (hybrid, extractive answers).

Example:
    python evaluation/build_eval_rows.py            # writes eval_rows.json

The committed ``evaluation/notebooks/error_analysis.ipynb`` reads the
written artifact; ``docs/rag/error-analysis.md`` narrates the findings.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import evaluate_generation  # noqa: E402
from src.config import RAGConfig  # noqa: E402
from src.embedder import create_embedder  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

RESULTS = Path(__file__).parent / "results"
MODES = ("dense", "sparse", "hybrid", "hybrid+rerank")


def build_pipelines(config: RAGConfig, embedder) -> tuple[dict[str, RAGPipeline], RAGConfig]:
    """One shared index, one pipeline per mode — mirrors ``run_comparison``.

    Each pipeline gets a deep-copied config: ``RAGPipeline._retrieve``
    dispatches on ``config.retrieval_mode`` at *call* time while the
    retriever type is bound at construction, so the two must match per
    instance.
    """
    builder = RAGPipeline(config, embedder=embedder, llm=None)
    builder.build_index()
    store = builder.store

    pipelines: dict[str, RAGPipeline] = {}
    for key, mode, rerank in (("dense", "dense", False), ("sparse", "sparse", False),
                              ("hybrid", "hybrid", False), ("hybrid+rerank", "hybrid", True)):
        cfg = copy.deepcopy(config)
        cfg.retrieval_mode = mode
        cfg.reranker = "lexical" if rerank else "none"
        pipeline = RAGPipeline(cfg, embedder=embedder, llm=None)
        pipeline.store = store
        pipeline.retriever.store = store
        pipelines[key] = pipeline
    return pipelines, builder.config


def main() -> None:
    config = RAGConfig()
    config.embedding_backend = "fake"
    embedder = create_embedder(
        config.embedding_backend, config.embedding_model,
        dimension=config.embedding_dimension,
    )
    pairs = json.loads((Path(__file__).parent / "qa_pairs.json").read_text("utf-8"))["pairs"]

    pipelines, config = build_pipelines(config, embedder)

    per_question: list[dict] = []
    for pair in pairs:
        row: dict = {"question": pair["question"], "expected_doc": pair["expected_doc"]}
        for key in MODES:
            results = pipelines[key].retrieve(pair["question"], top_k=3)
            docs = [r.chunk.doc_id + ".txt" for r in results]
            rank = docs.index(pair["expected_doc"]) + 1 if pair["expected_doc"] in docs else None
            row[key] = {"rank": rank, "top1_doc": docs[0] if docs else None}
        per_question.append(row)

    gen = evaluate_generation(pairs, pipelines["hybrid"], top_k=config.top_k, embedder=embedder)
    gen_rows = gen.pop("rows")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "eval_rows.json"
    out.write_text(
        json.dumps({
            "n_questions": len(pairs),
            "backend": "fake",
            "retrieval_ranks": per_question,
            "generation_rows": gen_rows,
            "generation_aggregate": gen,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[rows] {len(per_question)} questions -> {out}")
    for key in MODES:
        h1 = sum(1 for r in per_question if r[key]["rank"] == 1) / len(per_question)
        h3 = sum(1 for r in per_question if r[key]["rank"] is not None) / len(per_question)
        print(f"  {key:15} hit@1={h1:.2f}  hit@3={h3:.2f}")


if __name__ == "__main__":
    main()
