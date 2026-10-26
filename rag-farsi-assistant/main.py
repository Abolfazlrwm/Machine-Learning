"""Command line interface for the Persian RAG assistant.

Examples:
    python main.py build
    python main.py ask "بیش‌برازش چیست؟"
    python main.py chat
    python main.py info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig  # noqa: E402
from src.llm import create_provider  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def make_pipeline(args: argparse.Namespace) -> RAGPipeline:
    config = RAGConfig()
    if args.backend:
        config.embedding_backend = args.backend
    if args.provider:
        config.provider = args.provider
    config.top_k = args.top_k
    embedder_backend = config.embedding_backend
    from src.embedder import create_embedder

    embedder = create_embedder(
        embedder_backend, config.embedding_model, dimension=config.embedding_dimension
    )
    llm = None
    if not args.no_llm:
        llm = create_provider(config)
        print(f"[llm] provider={llm.name}")
    pipeline = RAGPipeline(config, embedder=embedder, llm=llm)
    return pipeline


def cmd_build(args: argparse.Namespace) -> None:
    pipeline = make_pipeline(args)
    stats = pipeline.build_index()
    print(f"[index] documents={stats['documents']} chunks={stats['chunks']}")
    print(f"[index] saved to {pipeline.config.index_dir}")


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = make_pipeline(args)
    pipeline.ensure_index()
    result = pipeline.answer(args.question, top_k=args.top_k)
    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for i, src in enumerate(result.sources, start=1):
        print(f"  [{i}] {src.title} (score={src.score}) :: {src.snippet[:80]}…")


def cmd_chat(args: argparse.Namespace) -> None:
    pipeline = make_pipeline(args)
    pipeline.ensure_index()
    print("دستیار فارسی RAG — برای خروج exit یا quit را بنویس.\n")
    while True:
        try:
            question = input("سؤال: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nخداحافظ!")
            break
        if question.lower() in {"exit", "quit", "خروج"}:
            print("خداحافظ!")
            break
        if not question:
            continue
        result = pipeline.answer(question)
        print(f"\n{result.answer}\n")
        print("منابع: " + ", ".join(f"{s.title} ({s.score})" for s in result.sources) + "\n")


def cmd_info(args: argparse.Namespace) -> None:
    pipeline = make_pipeline(args)
    if not pipeline.load_index():
        print("هیچ ایندکس ذخیره‌شده‌ای نیست؛ اول `python main.py build` را اجرا کن.")
        return
    print(pipeline.dump_stats())


def main() -> None:
    parser = argparse.ArgumentParser(description="Persian RAG assistant CLI")
    parser.add_argument("--provider", choices=["ollama", "openai"], default=None)
    parser.add_argument("--backend", choices=["sentence-transformers", "fake"], default=None)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-llm", action="store_true",
                        help="retrieval only, skip answer generation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="build and save the vector index").set_defaults(func=cmd_build)
    ask = sub.add_parser("ask", help="answer a single question")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)
    sub.add_parser("chat", help="interactive chat loop").set_defaults(func=cmd_chat)
    sub.add_parser("info", help="show index stats").set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
