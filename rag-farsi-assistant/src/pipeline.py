"""End-to-end RAG pipeline: index building, retrieval, grounded generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_documents
from .config import RAGConfig
from .data_loader import load_documents
from .embedder import BaseEmbedder, create_embedder
from .hybrid import HybridRetriever
from .llm import LLMProvider, create_provider
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever, RetrievalResult
from .rerank import create_reranker
from .vector_store import VectorStore


@dataclass
class Source:
    title: str
    chunk_id: str
    score: float
    snippet: str


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)


class RAGPipeline:
    """Facade tying all components together.

    Components are injectable (for tests / custom setups); by default they
    are created from :class:`RAGConfig`.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedder: BaseEmbedder | None = None,
        store: VectorStore | None = None,
        llm: LLMProvider | None = None,
    ):
        self.config = config
        self.config.ensure_dirs()
        self.embedder = embedder or create_embedder(
            config.embedding_backend,
            config.embedding_model,
            dimension=config.embedding_dimension,
        )
        self.store = store or VectorStore(dimension=self.embedder.dimension)
        self.llm = llm
        reranker = create_reranker(config.reranker) if config.reranker != "none" else None
        if config.retrieval_mode == "dense":
            self.retriever = Retriever(
                embedder=self.embedder,
                store=self.store,
                top_k=config.top_k,
                mmr_lambda=config.mmr_lambda,
                candidate_multiplier=config.candidate_multiplier,
            )
        else:
            # "sparse" (BM25 only) or "hybrid" (BM25 + dense via RRF)
            self.retriever = HybridRetriever(
                embedder=self.embedder,
                store=self.store,
                top_k=config.top_k,
                mode=config.retrieval_mode,
                rrf_k=config.rrf_k,
                candidate_multiplier=config.candidate_multiplier,
                mmr_lambda=config.mmr_lambda,
                bm25_k1=config.bm25_k1,
                bm25_b=config.bm25_b,
                reranker=reranker,
            )

    # --------------------------------------------------------------- index
    def build_index(self) -> dict:
        """Load documents -> chunk -> embed -> store. Returns index stats."""
        documents = load_documents(self.config.data_dir)
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        store = VectorStore(dimension=self.embedder.dimension)
        batch = 64
        for start in range(0, len(chunks), batch):
            batch_chunks = chunks[start:start + batch]
            vectors = self.embedder.embed([c.text for c in batch_chunks])
            store.add(vectors, batch_chunks)
        store.save(self.config.index_dir)
        self.store = store
        self.retriever.store = store
        return {"documents": len(documents), "chunks": len(chunks)}

    def load_index(self) -> bool:
        """Load a previously built index. Returns False if none exists."""
        meta_file = self.config.index_dir / "meta.json"
        if not meta_file.exists():
            return False
        self.store = VectorStore.load(self.config.index_dir)
        self.retriever.store = self.store
        return True

    def ensure_index(self) -> None:
        if len(self.store) == 0:
            if not self.load_index():
                self.build_index()

    # ------------------------------------------------------------- answers
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.ensure_index()
        return self.retriever.retrieve(question, top_k=top_k)

    def answer(self, question: str, top_k: int | None = None) -> AnswerResult:
        """Retrieve -> build prompt -> generate (or fall back to extractive)."""
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return AnswerResult(question=question, answer=self._no_index_answer())

        chunks = [r.chunk for r in results]
        sources = [
            Source(
                title=r.chunk.title,
                chunk_id=r.chunk.chunk_id,
                score=round(r.score, 4),
                snippet=self._snippet(r.chunk.text),
            )
            for r in results
        ]
        answer_text = self._generate(question, chunks)
        return AnswerResult(question=question, answer=answer_text, sources=sources)

    # -------------------------------------------------------------- helpers
    def _generate(self, question: str, chunks: list[Chunk]) -> str:
        if self.llm is None:
            # No LLM configured -> extractive fallback so the pipeline still
            # works end-to-end (useful for tests and retrieval-only runs).
            return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
        user_prompt = build_user_prompt(question, chunks)
        return self.llm.generate(
            SYSTEM_PROMPT, user_prompt, temperature=self.config.temperature
        )

    def _no_index_answer(self) -> str:
        return "هیچ سندی در ایندکس موجود نیست؛ اول دستور build را اجرا کن."

    def _snippet(self, text: str) -> str:
        limit = self.config.snippet_chars
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    # -------------------------------------------------------------- meta
    def stats(self) -> dict:
        docs = {c.doc_id for c in self.store.all_chunks()}
        return {"documents": len(docs), "chunks": len(self.store),
                "backend": self.embedder.name}

    def dump_stats(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False, indent=2)
