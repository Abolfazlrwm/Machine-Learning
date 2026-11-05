"""Streamlit chat UI for the Persian RAG assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig  # noqa: E402
from src.embedder import create_embedder  # noqa: E402
from src.llm import create_provider  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="دستیار RAG فارسی", page_icon="📚", layout="centered")

# RTL styling for Persian chat
st.markdown(
    """
    <style>
      .stChatMessage { direction: rtl; text-align: right; }
      [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="در حال آماده‌سازی ایندکس…")
def build_pipeline(provider: str, model: str, top_k: int, mmr_lambda: float,
                   backend: str, embedding_model: str, retrieval_mode: str) -> RAGPipeline:
    config = RAGConfig()
    config.provider = provider
    config.embedding_backend = backend
    config.embedding_model = embedding_model
    config.top_k = top_k
    config.mmr_lambda = mmr_lambda
    config.retrieval_mode = retrieval_mode
    if provider == "ollama":
        config.ollama_model = model
    else:
        config.openai_model = model
    embedder = create_embedder(backend, embedding_model, config.embedding_dimension)
    llm = create_provider(config)
    pipeline = RAGPipeline(config, embedder=embedder, llm=llm)
    pipeline.ensure_index()
    return pipeline


with st.sidebar:
    st.header("⚙️ تنظیمات")
    provider = st.radio("LLM", ["ollama", "openai"], help="Ollama محلی و رایگان است؛ openai با هر سرویس سازگار با OpenAI کار می‌کند (OpenRouter/Groq/…)")
    default_model = "llama3.1" if provider == "ollama" else "gpt-4o-mini"
    model = st.text_input("نام مدل", value=default_model)
    top_k = st.slider("تعداد قطعه‌های بازیابی (top_k)", 1, 8, 4)
    mmr_lambda = st.slider("تنوع MMR (λ)", 0.0, 1.0, 0.7, 0.1)
    retrieval_mode = st.radio(
        "حالت بازیابی",
        ["hybrid", "dense", "sparse"],
        help="hybrid = ترکیب بردار معنایی + BM25 با RRF (پیشنهاد‌شده)، dense فقط Embedding، sparse فقط BM25",
    )
    backend = st.selectbox(
        "بک‌اند Embedding",
        ["sentence-transformers", "fake"],
        help="«fake» بدون دانلود مدل کار می‌کند (فقط برای تست خط لوله)",
    )
    embedding_model = st.text_input(
        "مدل Embedding",
        value="intfloat/multilingual-e5-base",
        disabled=(backend == "fake"),
    )
    rebuild = st.button("🔄 بازسازی ایندکس")
    st.caption("سؤال‌ها فقط از روی اسناد پوشه‌ی data پاسخ داده می‌شوند.")

try:
    if rebuild:
        build_pipeline.clear()
    pipeline = build_pipeline(provider, model, top_k, mmr_lambda, backend, embedding_model, retrieval_mode)
    stats = pipeline.stats()
    st.sidebar.success(f"آماده است: {stats['chunks']} قطعه از {stats['documents']} سند")
except Exception as exc:  # pragma: no cover - UI error path
    st.error(f"خطا در آماده‌سازی: {exc}")
    if provider == "ollama":
        st.info("اگر Ollama نصب نیست: `ollama pull llama3.1` و سپس `ollama serve` را اجرا کن. تا آن موقع می‌توانی در تنظیمات provider را روی openai بگذاری.")
    st.stop()

st.title("📚 دستیار پرسش‌وپاسخ فارسی")
st.caption("RAG روی اسناد مفاهیم یادگیری ماشین — پاسخ‌ها با ارجاع به منبع")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("منابع استفاده‌شده"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}] {src['title']}** — امتیاز: `{src['score']}`")
                    st.markdown(f"> {src['snippet']}")

if question := st.chat_input("سؤالت را بپرس…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("در حال بازیابی اسناد مرتبط…"):
            try:
                result = pipeline.answer(question, top_k=top_k)
            except Exception as exc:
                st.error(f"خطا در تولید پاسخ: {exc}")
                st.stop()
        st.markdown(result.answer)
        sources_payload = [
            {"title": s.title, "score": s.score, "snippet": s.snippet}
            for s in result.sources
        ]
        with st.expander("منابع استفاده‌شده"):
            for i, src in enumerate(sources_payload, start=1):
                st.markdown(f"**[{i}] {src['title']}** — امتیاز: `{src['score']}`")
                st.markdown(f"> {src['snippet']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": sources_payload}
    )
