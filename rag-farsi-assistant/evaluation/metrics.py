"""RAGAS-style evaluation metrics — deterministic, dependency-free proxies.

The original RAGAS suite uses an LLM as a judge. The metrics here implement
the same four dimensions with transparent token-overlap heuristics so the
evaluation runs anywhere (CI, offline) and stays fully reproducible:

* ``context_precision`` — are the retrieved chunks on-topic? (AP-style,
  a chunk counts as relevant when its token-F1 with the gold answer ≥ τ)
* ``context_recall``    — how much of the gold answer's vocabulary do the
  retrieved chunks cover?
* ``faithfulness``      — what fraction of the answer's sentences are
  grounded in the retrieved contexts?
* ``answer_relevancy``  — how close is the answer to the question? Uses an
  embedder cosine when an embedder is supplied, token-F1 otherwise.

When the pipeline runs with a real LLM provider (Ollama / OpenAI) these are
honest, if conservative, quality estimates; with the extractive fallback
answers they double as an end-to-end smoke check.
"""
from __future__ import annotations

import re

import numpy as np

from src.bm25 import tokenize

_SENT_RE = re.compile(r"[^.!؟?\n]+[.!؟?\n]*")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that understands Persian enders (؟ ؟ … and newlines)."""
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def token_f1(a: str, b: str) -> float:
    """F1 between the token sets of two texts (precision ∩ recall harmonic)."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if not overlap:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def _coverage(numerator: set[str], denominator: set[str]) -> float:
    if not denominator:
        return 0.0
    return len(numerator & denominator) / len(denominator)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def context_precision(
    retrieved_texts: list[str], gold_answer: str, tau: float = 0.2
) -> float:
    """Average precision over the retrieved chunks.

    A chunk is *relevant* when ``token_f1(chunk, gold_answer) >= tau``.
    Relevant chunks ranked early score higher, mirroring RAGAS.
    """
    if not retrieved_texts:
        return 0.0
    relevant = 0
    score = 0.0
    for i, text in enumerate(retrieved_texts, start=1):
        if token_f1(text, gold_answer) >= tau:
            relevant += 1
            score += relevant / i
    return score / relevant if relevant else 0.0


def context_recall(retrieved_texts: list[str], gold_answer: str) -> float:
    """Fraction of gold-answer tokens covered by the retrieved contexts."""
    gold = _token_set(gold_answer)
    if not gold:
        return 0.0
    union: set[str] = set()
    for text in retrieved_texts:
        union |= _token_set(text)
    return _coverage(union, gold)


def faithfulness(answer: str, contexts: list[str], tau: float = 0.5) -> float:
    """Fraction of answer sentences whose tokens are grounded in the contexts.

    A sentence is *supported* when at least ``tau`` of its (non-stopword)
    tokens appear in the union of the contexts.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0
    union: set[str] = set()
    for text in contexts:
        union |= _token_set(text)
    supported = 0
    for sentence in sentences:
        tokens = _token_set(sentence)
        if not tokens or _coverage(union, tokens) >= tau:
            supported += 1
    return supported / len(sentences)


def answer_relevancy(question: str, answer: str, embedder=None) -> float:
    """Semantic closeness of the answer to the original question.

    With an ``embedder`` (E5 in production): cosine between the question and
    the answer embeddings. Without one: token-F1 fallback, which is crude
    but deterministic and dependency-free.
    """
    if embedder is not None:
        q = np.asarray(embedder.embed_query(question), dtype="float32").reshape(-1)
        a = np.asarray(embedder.embed([answer])[0], dtype="float32").reshape(-1)
        denom = float(np.linalg.norm(q) * np.linalg.norm(a))
        return float(np.dot(q, a) / denom) if denom else 0.0
    return token_f1(question, answer)


# ---------------------------------------------------------------------------
# Full generation evaluation
# ---------------------------------------------------------------------------


def evaluate_generation(
    pairs: list[dict],
    pipeline,  # RAGPipeline (llm=None -> extractive answers)
    top_k: int = 4,
    embedder=None,
) -> dict:
    """Run the full RAGAS-style suite over golden QA pairs.

    Returns aggregate metrics plus one row per question so failures can be
    inspected individually. ``answer_relevancy`` is only meaningful for
    generative answers — with the extractive fallback (no LLM) it is set to
    ``None`` instead of a misleading token-overlap value.
    """
    rows: list[dict] = []
    generative = getattr(pipeline, "llm", None) is not None
    for pair in pairs:
        question = pair["question"]
        results = pipeline.retrieve(question, top_k=top_k)
        contexts = [r.chunk.text for r in results]
        retrieved_docs = [r.chunk.doc_id + ".txt" for r in results]
        answer = pipeline.answer(question, top_k=top_k).answer

        rows.append({
            "question": question,
            "expected_doc": pair["expected_doc"],
            "hit": pair["expected_doc"] in retrieved_docs,
            "context_precision": context_precision(contexts, pair["gold_answer"]),
            "context_recall": context_recall(contexts, pair["gold_answer"]),
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevancy": (
                answer_relevancy(question, answer, embedder) if generative else None
            ),
        })

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "n_questions": len(rows),
        "doc_hit_rate": mean("hit"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "rows": rows,
    }
