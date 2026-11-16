---
title: تولید متن فارسی با GPT از صفر — در مرورگر شما
emoji: 🧠
colorFrom: blue
colorTo: pink
sdk: static
pinned: true
license: mit
short_description: TinyGPT — autograd + transformer با NumPy، در WebAssembly
---

# 🧠 TinyGPT — تولید متن فارسی، بدون هیچ فریم‌ورکی و بدون سرور

دموی زندهٔ پروژهٔ
[`tiny-gpt-from-scratch`](https://github.com/Abolfazlrwm/Machine-Learning/tree/main/tiny-gpt-from-scratch):
یک موتور **autograd** دست‌ساز، یک ترنسفورمر decoder-only و بهینه‌ساز Adam —
**همه با NumPy خام، بدون PyTorch و بدون JAX** — که روی متن فارسی (همان
corpus دستیار RAG این ریپو، ۹٬۶۶۵ کاراکتر) آموزش دیده است.

این دمو یک **Static Space** است: Python (با Pyodide) و NumPy داخل مرورگر
اجرا می‌شوند و همان کد `src/` پروژه — بدون هیچ تغییری — از چک‌پوینت
۲ مگابایتی (`model/params.npz`) نتیجه می‌گیرد. هیچ سرور inferenceای در کار نیست.

* ۲ بلوک · d=96 · ۳ سر توجه · ۲۴۹٬۳۱۶ پارامتر · ۶۵ ثانیه آموزش روی CPU
* هر primitive در تست‌ها در برابر **گرادیان عددی** راستی‌آزمایی شده است

متن تولیدی در سطح «حروف» است (مدل char-level روی دادهٔ خیلی کم است) —
شروع کلمه‌ها، نیم‌فاصله و ریتم نشانه‌گذاری را یاد گرفته ولی روان نیست؛
خودِ این محدودیت بخشی از داستان پروژه است.
