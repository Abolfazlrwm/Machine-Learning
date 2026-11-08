"""Persian RAG demo — retrieval-grounded Q&A over six ML study notes.

Standalone, dependency-light demo of the hybrid retrieval idea from the
`rag-farsi-assistant` project (https://github.com/Abolfazlrwm/Machine-Learning):

* multilingual E5 embeddings for dense semantic retrieval (numpy top-k,
  no FAISS needed at this corpus size),
* a compact lexical (BM25-style) scorer as an automatic fallback,
* extractive, citation-carrying answers — the full generated pipeline
  (Ollama / OpenAI + RRF fusion + optional cross-encoder re-ranking)
  lives in the repository.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
EMBED_MODEL = os.environ.get("E5_MODEL", "intfloat/multilingual-e5-base")
TOP_K = 3
CHUNK_CHARS, CHUNK_OVERLAP = 700, 150

STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن",
    "یک", "های", "می", "شد", "کند", "می‌شود", "the", "a", "an", "of", "and",
}


# --------------------------------------------------------------- text utils
def normalise(text: str) -> str:
    """Arabic/Persian character + ZWNJ normalisation (same rules as the repo)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u064a", "\u06cc")   # Arabic Yeh -> Persian Yeh
    text = text.replace("\u0643", "\u06a9")   # Arabic Keh -> Persian Keh
    return text


def tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[\w\u0600-\u06FF]+", normalise(text).lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def chunk_documents(docs: list[tuple[str, str]]) -> list[dict]:
    chunks = []
    for title, text in docs:
        text = normalise(text)
        step = CHUNK_CHARS - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            part = text[i:i + CHUNK_CHARS].strip()
            if len(part) < 80:
                break
            chunks.append({
                "title": title,
                "chunk_id": f"{title}:{len(chunks)}",
                "text": part,
            })
    return chunks


# ------------------------------------------------------------- backends
class DenseRetriever:
    """E5 embeddings + cosine similarity (vectors are L2-normalised)."""

    name = "dense-e5"

    def __init__(self, chunks: list[dict]):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = chunks
        self.matrix = self.model.encode(
            [f"passage: {c['text']}" for c in chunks],
            batch_size=32, convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")
        self.matrix /= np.linalg.norm(self.matrix, axis=1, keepdims=True)

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        q = self.model.encode(
            [f"query: {question}"], convert_to_numpy=True
        )[0].astype("float32")
        q /= max(np.linalg.norm(q), 1e-9)
        scores = self.matrix @ q
        return top_chunks(self.chunks, scores, top_k)


class LexicalRetriever:
    """BM25-style fallback (idf-weighted bag of words) if E5 cannot load."""

    name = "lexical-bm25-fallback"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tok = [tokenize(c["text"]) for c in chunks]
        self.n = len(chunks)
        df: dict[str, int] = {}
        for toks in self.tok:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: np.log((self.n - d + 0.5) / (d + 0.5) + 1)
                    for t, d in df.items()}

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        q = tokenize(question)
        scores = np.zeros(self.n)
        for i, toks in enumerate(self.tok):
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            scores[i] = sum(
                self.idf.get(t, 0.0) * (3.0 * tf[t] / (tf[t] + 1.2))
                for t in q if t in tf
            )
        return top_chunks(self.chunks, scores, top_k)


def top_chunks(chunks: list[dict], scores: np.ndarray, top_k: int) -> list[dict]:
    """Indices of the *top_k* highest scores, annotated onto the chunks."""
    idx = np.argsort(-scores)[:top_k]
    return [{**chunks[i], "score": float(scores[i])} for i in idx]


# ----------------------------------------------------------------- answer
def load_corpus() -> list[tuple[str, str]]:
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        docs.append((path.stem, path.read_text(encoding="utf-8")))
    return docs


print("Loading corpus and building the retrieval index ...")
CHUNKS = chunk_documents(load_corpus())
try:
    RETRIEVER = DenseRetriever(CHUNKS)
    print(f"Dense index ready: {len(CHUNKS)} chunks, model={EMBED_MODEL}")
except Exception as exc:  # pragma: no cover - fallback for cold/offline builds
    print(f"E5 unavailable ({exc}); using the lexical fallback.")
    RETRIEVER = LexicalRetriever(CHUNKS)


def extractive_answer(question: str, hits: list[dict]) -> str:
    """Best query-overlapping sentences per source, with [n] citations."""
    q_tokens = set(tokenize(question))
    parts = ["پاسخ بر پایه‌ی منابع بازیابی‌شده:", ""]
    for rank, hit in enumerate(hits, 1):
        sentences = re.split(r"(?<=[.!؟])\s+|\n+", hit["text"])
        scored = []
        for s in sentences:
            overlap = len(q_tokens & set(tokenize(s)))
            if overlap:
                scored.append((overlap, s.strip()))
        scored.sort(key=lambda pair: -pair[0])
        best = " ".join(s for _, s in scored[:2]) or hit["text"][:200]
        parts.append(f"[{rank}] {best}")
        parts.append("")
    parts.append("برای پاسخِ تولیدشده با مدل زبانی، نسخهٔ کامل مخزن را ببینید.")
    return "\n".join(parts)


def answer_question(question: str):
    if not question.strip():
        return "یک پرسش بنویس 🙂", []
    hits = RETRIEVER.search(question)
    answer = extractive_answer(question, hits)
    sources = [
        [f"[{i}]", h["title"], f"{h['score']:.3f}", h["text"][:160] + "…"]
        for i, h in enumerate(hits, 1)
    ]
    return answer, sources


# --------------------------------------------------------------------- UI
with gr.Blocks(
    title="دستیار پرسش‌وپاسخ فارسی — RAG",
    css="#answer textarea {direction: rtl; text-align: right;} "
        ".gradio-container {max-width: 880px; margin: auto;}",
) as demo:
    gr.Markdown(
        f"""
# 🤖 دستیار پرسش‌وپاسخ فارسی (RAG)
بازیابی معنایی با مدل چندزبانه **E5** روی شش یادداشت آموزشی یادگیری ماشین
({len(CHUNKS)} قطعه). پاسخ‌ها **استخراجی و همراه ارجاع** هستند — نسخهٔ کامل
با جست‌وجوی ترکیبی BM25+RRF، رتبه‌بند مجدد و تولید پاسخ در
[مخزن گیت‌هاب](https://github.com/Abolfazlrwm/Machine-Learning) موجود است.
        """
    )
    with gr.Row():
        question = gr.Textbox(
            label="پرسش خود را بنویسید",
            placeholder="مثلاً: بیش‌برازش چیست و چطور جلوی آن را بگیریم؟",
            lines=2,
        )
    with gr.Row():
        btn = gr.Button("پاسخ بده", variant="primary")
    answer = gr.Textbox(label="پاسخ (استخراجی + ارجاع)", lines=10, elem_id="answer")
    sources = gr.Dataframe(
        headers=["ارجاع", "سند", "امتیاز", "بخش بازیابی‌شده"],
        label="منابع بازیابی‌شده",
        wrap=True,
    )
    btn.click(answer_question, inputs=question, outputs=[answer, sources])
    question.submit(answer_question, inputs=question, outputs=[answer, sources])
    gr.Examples(
        examples=[
            "بیش‌برازش (overfitting) چیست و چطور جلوی آن را گرفت؟",
            "تفاوت یادگیری نظارت‌شده و بدون نظارت چیست؟",
            "attention در ترنسفورمرها چطور کار می‌کند؟",
            "رانش داده (data drift) چیست و چطور با آن کنار بیاییم؟",
        ],
        inputs=question,
    )

if __name__ == "__main__":
    demo.launch()
