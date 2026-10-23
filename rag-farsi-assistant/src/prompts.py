"""Persian prompt templates for grounded generation.

The prompt is intentionally strict: the model must answer *only* from the
retrieved context, cite sources as [1], [2], ... and admit when the
context is insufficient — the core anti-hallucination pattern in RAG.
"""
from __future__ import annotations

from .chunker import Chunk

SYSTEM_PROMPT = (
    "تو یک دستیار پاسخگوی دقیق هستی که فقط بر اساس «قطعه‌های بازیابی‌شده» "
    "پاسخ می‌دهد. قواعد:\n"
    "1) فقط از اطلاعات موجود در قطعه‌ها استفاده کن و از دانش بیرونی و حدس زدن پرهیز کن.\n"
    "2) پاسخ را به زبان فارسی، روان و کوتاه بنویس.\n"
    "3) کنار هر ادعا، شماره منبع را داخل کروشه بنویس؛ مثل [1] یا [2].\n"
    "4) اگر اطلاعات کافی برای پاسخ در قطعه‌ها نبود، صریح بگو: "
    "«با اسناد موجود نمی‌توانم به این پرسش پاسخ دهم.»"
)


def build_context_block(chunks: list[Chunk]) -> str:
    """Format retrieved chunks as a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (منبع: {chunk.title})\n{chunk.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    context = build_context_block(chunks)
    return (
        "قطعه‌های بازیابی‌شده از اسناد:\n\n"
        f"{context}\n\n"
        "---\n"
        f"پرسش کاربر: {question}\n\n"
        "پاسخ (با ارجاع به شماره منابع):"
    )
