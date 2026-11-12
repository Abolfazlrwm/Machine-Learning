"""Retrieval & generation evaluation for the Persian RAG assistant.

Retrieval quality  : Hit@k, MRR and nDCG@k over the golden QA pairs (no LLM).
Mode comparison    : dense (embeddings+MMR) vs sparse (BM25) vs hybrid (RRF),
                     plus ``hybrid+rerank`` — the same hybrid results passed
                     through the lexical re-ranker (``--no-rerank`` to skip).
Generation quality : RAGAS-style metrics from ``evaluation/metrics.py`` —
                     context precision/recall, faithfulness, answer relevancy.

Examples:
    python evaluate.py                                  # dense, Hit@k + MRR
    python evaluate.py --backend fake                   # dependency-free run
    python evaluate.py --compare                        # all modes + rerank
    python evaluate.py --mode hybrid --generation       # + RAGAS-style suite
    python evaluate.py --compare --generation --out evaluation/results/report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.metrics import ndcg_at_k  # noqa: E402
from src.config import RAGConfig  # noqa: E402
from src.embedder import create_embedder  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

EVAL_DIR = Path(__file__).parent / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
MODES = ("dense", "sparse", "hybrid")
RERANK_MODE = "hybrid+rerank"


def load_pairs() -> list[dict]:
    return json.loads((EVAL_DIR / "qa_pairs.json").read_text(encoding="utf-8"))["pairs"]


def evaluate(pipeline: RAGPipeline, ks: tuple[int, ...] = (1, 2, 3)) -> dict:
    pairs = load_pairs()
    hits = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    ndcg_sums = {k: 0.0 for k in ks}

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
            ndcg_sums[k] += ndcg_at_k(ranked_docs, expected, k)

    n = len(pairs)
    return {
        "n_questions": n,
        **{f"hit@{k}": round(hits[k] / n, 4) for k in ks},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        **{f"ndcg@{k}": round(ndcg_sums[k] / n, 4) for k in ks},
    }


def make_pipeline(config: RAGConfig, embedder, llm=None) -> RAGPipeline:
    return RAGPipeline(config, embedder=embedder, llm=llm)


def run_comparison(
    config: RAGConfig, embedder, rebuild: bool = False, rerank: bool = True
) -> dict[str, dict]:
    """Evaluate every retrieval mode against the same golden QA set.

    When ``rerank`` is set, a final ``hybrid+rerank`` column runs the same
    hybrid retrieval through the lexical re-ranker — the offline-safe way
    to show what the re-rank stage does to the ranking.
    """
    per_mode: dict[str, dict] = {}
    shared_store = None
    for mode in MODES:
        config.retrieval_mode = mode
        pipeline = make_pipeline(config, embedder)
        if rebuild or len(pipeline.store) == 0 or not pipeline.load_index():
            pipeline.build_index()
        # reuse the built index across modes (identical chunks & embeddings)
        if mode == MODES[0]:
            shared_store = pipeline.store
        else:
            pipeline.store = shared_store
            pipeline.retriever.store = shared_store
        per_mode[mode] = evaluate(pipeline)

    if rerank:
        config.retrieval_mode = "hybrid"
        config.reranker = "lexical"
        pipeline = make_pipeline(config, embedder)
        pipeline.store = shared_store
        pipeline.retriever.store = shared_store
        per_mode[RERANK_MODE] = evaluate(pipeline)
        config.reranker = "none"
    config.retrieval_mode = "hybrid"
    return per_mode


def format_comparison_md(per_mode: dict[str, dict]) -> str:
    modes = list(per_mode)
    metrics = ["hit@1", "hit@2", "hit@3", "mrr", "ndcg@3"]
    header = "| metric | " + " | ".join(modes + ["best"]) + " |"
    sep = "|---" * (len(modes) + 2) + "|"
    lines = [header, sep]
    for m in metrics:
        values = {mode: per_mode[mode][m] for mode in modes}
        best = max(values, key=values.get)
        cells = " | ".join(
            f"**{v}**" if mode == best else str(v) for mode, v in values.items()
        )
        lines.append(f"| {m} | {cells} | {best} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation quality")
    parser.add_argument("--backend", choices=["sentence-transformers", "fake"], default="sentence-transformers")
    parser.add_argument("--rebuild", action="store_true", help="rebuild index before evaluating")
    parser.add_argument("--mode", choices=MODES, default="dense", help="retrieval mode to evaluate")
    parser.add_argument("--compare", action="store_true", help="evaluate dense vs sparse vs hybrid (+ rerank)")
    parser.add_argument(
        "--no-rerank", dest="rerank", action="store_false",
        help="skip the hybrid+rerank column in --compare",
    )
    parser.add_argument("--generation", action="store_true", help="run the RAGAS-style generation suite")
    parser.add_argument("--out", default=None, help="write a Markdown + JSON report here (stem path)")
    args = parser.parse_args()

    config = RAGConfig()
    config.embedding_backend = args.backend
    embedder = create_embedder(
        config.embedding_backend, config.embedding_model, dimension=config.embedding_dimension
    )

    report_parts: list[str] = [
        "# Retrieval & Generation Evaluation",
        "",
        f"Backend: `{args.backend}` · corpus: `data/` · golden set: `{len(load_pairs())} QA pairs`",
        "",
    ]
    report_json: dict = {"backend": args.backend}

    if args.compare:
        per_mode = run_comparison(config, embedder, rebuild=args.rebuild, rerank=args.rerank)
        print(json.dumps(per_mode, ensure_ascii=False, indent=2))
        table = format_comparison_md(per_mode)
        note = (
            "`hybrid+rerank` = hybrid results re-ordered by the lexical "
            "re-ranker (`src/rerank.py`), same index, no extra retrieval."
            if RERANK_MODE in per_mode
            else ""
        )
        report_parts += ["## Retrieval modes compared (Hit@k / MRR / nDCG@k)", "", table, ""]
        if note:
            report_parts += [note, ""]
        report_json["retrieval_by_mode"] = per_mode
    else:
        config.retrieval_mode = args.mode
        pipeline = make_pipeline(config, embedder)
        if args.rebuild or not pipeline.load_index():
            stats = pipeline.build_index()
            print(f"[index] rebuilt: {stats}")
        metrics = evaluate(pipeline)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        report_parts += [f"## Retrieval ({args.mode})", "", f"```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```", ""]
        report_json["retrieval"] = metrics

    if args.generation:
        from evaluation.metrics import evaluate_generation

        config.retrieval_mode = "hybrid"
        pipeline = make_pipeline(config, embedder, llm=None)  # extractive answers
        if len(pipeline.store) == 0 and not pipeline.load_index():
            pipeline.build_index()
        gen = evaluate_generation(
            load_pairs(), pipeline, top_k=config.top_k, embedder=embedder
        )
        aggregate = {k: v for k, v in gen.items() if k != "rows"}
        print(json.dumps(aggregate, ensure_ascii=False, indent=2))
        report_parts += [
            "## RAGAS-style generation metrics (extractive answer mode)",
            "",
            "Answers are produced by the deterministic extractive fallback, so",
            "`faithfulness` doubles as an end-to-end grounding smoke check; run",
            "with a live LLM provider for true generative scoring.",
            "",
            f"```json\n{json.dumps(aggregate, ensure_ascii=False, indent=2)}\n```",
            "",
        ]
        report_json["generation"] = aggregate

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text("\n".join(report_parts), encoding="utf-8")
        out.with_suffix(".json").write_text(
            json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[report] written to {out.with_suffix('.md')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
