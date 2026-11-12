"""End-to-end RAG pipeline: index building, retrieval, grounded generation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .cache import TTLCache
from .chunker import Chunk, chunk_documents
from .config import RAGConfig
from .data_loader import load_documents
from .embedder import BaseEmbedder, create_embedder
from .guardrails import Detection, InjectionDetector, REFUSAL_ANSWER
from .hybrid import HybridRetriever
from .llm import LLMProvider, create_provider
from .observability import Trace
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .query import QueryPlan, analyse_query
from .retriever import Retriever, RetrievalResult
from .rerank import create_reranker
from .semantic_cache import SemanticCache
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
    meta: dict = field(default_factory=dict)  # cache / timings / query plan


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
        # Answer cache: disabled while cache_ttl == 0 (the default keeps
        # evaluation runs deterministic; the serving layer enables it).
        # With `semantic_cache` on, the exact-match TTL cache is upgraded to
        # an embedding-similarity cache so paraphrased questions hit too.
        if config.cache_ttl > 0:
            if config.semantic_cache:
                self.cache: TTLCache | SemanticCache = SemanticCache(
                    embedder=self.embedder,
                    threshold=config.semantic_cache_threshold,
                    maxsize=config.cache_maxsize,
                    ttl=config.cache_ttl,
                )
            else:
                self.cache = TTLCache(
                    maxsize=config.cache_maxsize, ttl=config.cache_ttl
                )
        else:
            self.cache = None
        # Prompt-injection screening (src/guardrails.py): None = disabled.
        self._detector = (
            InjectionDetector(config.injection_threshold)
            if config.guardrails != "off"
            else None
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
    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        trace: Trace | None = None,
    ) -> list[RetrievalResult]:
        self.ensure_index()
        return self._retrieve(analyse_query(question), top_k, trace)

    def _retrieve(
        self, plan: QueryPlan, top_k: int | None, trace: Trace | None
    ) -> list[RetrievalResult]:
        """Run retrieval with the query plan applied (see src/query.py).

        Expansion terms go to the *lexical* sub-query only; the dense
        sub-query embeds the visitor's original wording.
        """
        if not plan.expansions or self.config.retrieval_mode == "dense":
            return self.retriever.retrieve(plan.original, top_k=top_k, trace=trace)
        return self.retriever.retrieve(
            plan.expanded, top_k=top_k, trace=trace, embed_text=plan.original
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        trace: Trace | None = None,
    ) -> AnswerResult:
        """Guardrail policy -> the full retrieve/prompt/generate path.

        With ``guardrails="flag"`` (the default) every question is screened
        and flagged attempts keep their answer but carry
        ``result.meta["injection"]``. With ``"block"`` a flagged question
        never reaches retrieval or the LLM — the caller receives
        :data:`~src.guardrails.REFUSAL_ANSWER` plus the detection report.
        See :mod:`src.guardrails` for the policy details.
        """
        detection: Detection | None = None
        if self._detector is not None:
            detection = self._detector.detect(question)
            if detection.flagged and self.config.guardrails == "block":
                tr = trace if trace is not None else Trace()
                return self._blocked_result(question, detection, tr)

        result = self._answer_impl(question, top_k, trace)
        if detection is not None and detection.flagged:
            result.meta["injection"] = detection.to_meta()
            result.meta["guardrails"] = "flag"
        return result

    def _blocked_result(
        self, question: str, detection: Detection, tr: Trace
    ) -> AnswerResult:
        """The up-front refusal returned when ``guardrails="block"``."""
        return AnswerResult(
            question=question,
            answer=REFUSAL_ANSWER,
            sources=[],
            meta={
                "cache": "off",
                "guardrails": "block",
                "injection": detection.to_meta(),
                "timings_ms": tr.timings_ms(),
            },
        )

    def _answer_impl(
        self,
        question: str,
        top_k: int | None = None,
        trace: Trace | None = None,
    ) -> AnswerResult:
        """Retrieve -> build prompt -> generate (or fall back to extractive).

        Pass a :class:`~src.observability.Trace` to collect stage timings;
        one is created (and returned via ``result.meta``) otherwise. When
        ``config.cache_ttl > 0`` repeated questions are served from the
        TTL/LRU cache and ``result.meta["cache"]`` says ``"hit"``.
        """
        tr = trace if trace is not None else Trace()
        plan = analyse_query(question)
        self.ensure_index()  # index must exist before the cache key sees it

        cache_key = None
        if self.cache is not None:
            if isinstance(self.cache, SemanticCache):
                cache_key = plan.normalised
                hit, cached, similarity = self.cache.lookup(
                    plan.normalised, namespace=self._cache_namespace(top_k)
                )
            else:
                cache_key = self._cache_key(plan, top_k)
                hit, cached = self.cache.get(cache_key)
                similarity = None
            if hit:
                tr.note("cache", "hit")
                if similarity is not None:
                    tr.note("cache_similarity", similarity)
                meta: dict = {"cache": "hit", "timings_ms": tr.timings_ms()}
                if similarity is not None:
                    meta["cache_similarity"] = round(similarity, 4)
                return AnswerResult(
                    question=question,
                    answer=cached[0],
                    sources=list(cached[1]),
                    meta=meta,
                )
            tr.note("cache", "miss")
        else:
            tr.note("cache", "off")

        with tr.span("retrieve"):
            results = self._retrieve(plan, top_k, tr)
        if not results:
            return AnswerResult(
                question=question,
                answer=self._no_index_answer(),
                meta=self._meta(tr, plan, cache_state=tr.notes.get("cache", "off")),
            )

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
        with tr.span("generate"):
            answer_text = self._generate(plan.original, chunks)

        result = AnswerResult(
            question=question,
            answer=answer_text,
            sources=sources,
            meta=self._meta(tr, plan, tr.notes.get("cache", "off")),
        )
        if self.cache is not None and cache_key:
            if isinstance(self.cache, SemanticCache):
                self.cache.put(
                    cache_key,
                    (answer_text, sources),
                    namespace=self._cache_namespace(top_k),
                )
            else:
                self.cache.put(cache_key, (answer_text, sources))
        return result

    # -------------------------------------------------------------- helpers
    def _meta(self, tr: Trace, plan: QueryPlan, cache_state: str) -> dict:
        """Observability block attached to every AnswerResult."""
        meta: dict = {
            "cache": cache_state,
            "timings_ms": tr.timings_ms(),
            "query": {
                "original": plan.original,
                "expansions": plan.expansions,
                "mixed_script": plan.is_mixed,
            },
        }
        if "cache_similarity" in tr.notes:
            meta["cache_similarity"] = round(tr.notes["cache_similarity"], 4)
        return meta

    def _cache_namespace(self, top_k: int | None) -> str:
        """The settings that change the answer for a given question."""
        return "|".join(
            [
                str(top_k),
                self.config.retrieval_mode,
                self.config.reranker,
                self.embedder.name,
                str(len(self.store)),
            ]
        )

    def _cache_key(self, plan: QueryPlan, top_k: int | None) -> str:
        """Cache identity: folded question + the settings that change output."""
        basis = "|".join([plan.normalised, self._cache_namespace(top_k)])
        return hashlib.sha1(basis.encode("utf-8"), usedforsecurity=False).hexdigest()

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
        out = {"documents": len(docs), "chunks": len(self.store),
               "backend": self.embedder.name}
        if self.cache is not None:
            out["cache"] = self.cache.stats
        return out

    def dump_stats(self) -> str:
        return json.dumps(self.stats(), ensure_ascii=False, indent=2)
