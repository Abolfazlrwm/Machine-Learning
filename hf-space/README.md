---
title: دستیار پرسش‌وپاسخ فارسی — RAG
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# دستیار پرسش‌وپاسخ فارسی (RAG) — نسخهٔ دمو

دموی زندهٔ ایدهٔ بازیابی ترکیبی پروژهٔ
[`rag-farsi-assistant`](https://github.com/Abolfazlrwm/Machine-Learning/tree/main/rag-farsi-assistant):

- **بازیابی معنایی** با مدل چندزبانهٔ `intfloat/multilingual-e5-base`
  (شش یادداشت آموزشی ML، قطعه‌بندی ۷۰۰ نویسه‌ای با هم‌پوشانی)
- **پاسخ استخراجی همراه ارجاع** `[1] [2]` — فقط از متن منابع بازیابی‌شده
- **fallback واژگانی** (BM25-style) اگر دانلود مدل در ساخت اولیه ممکن نشد

## اجرای همین‌جا (Hugging Face Spaces)

این پوشه دقیقاً همان چیزی است که باید در یک Space آپلود شود:

```bash
# ۱) یک Space با SDK گرادیو بسازید
# ۲) محتوای این پوشه را push کنید
git clone https://huggingface.co/spaces/<user>/<space-name>
cp -r hf-space/* <space-name>/ && cd <space-name>
git add . && git commit -m "feat: Persian RAG demo" && git push
```

## نسخهٔ کامل در مخزن

نسخهٔ کامل پروژه شامل موارد زیر است که در این دمو سبک‌سازی شده:

| قابلیت | این دمو | مخزن |
|---|---|---|
| بازیابی | E5 + cosine | **BM25 از صفر + RRF** + E5 + MMR |
| رتبه‌بند مجدد | — | cross-encoder اختیاری |
| تولید پاسخ | استخراجی | Ollama / OpenAI با پرامپت فارسی سخت‌گیرانه |
| ارزیابی | — | مجموعه طلایی ۳۰ pair + سنجه‌های RAGAS-style |
| سرویس‌دهی | گرادیو | **FastAPI + Docker** (`serving/`) |
