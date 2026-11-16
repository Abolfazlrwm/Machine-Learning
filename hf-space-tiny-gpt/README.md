---
title: تولید متن فارسی با GPT از صفر
emoji: 🧠
colorFrom: blue
colorTo: pink
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: TinyGPT — autograd + transformer، همه با NumPy خام
---

# 🧠 TinyGPT — تولید متن فارسی، بدون هیچ فریم‌ورکی

دموی زندهٔ پروژهٔ
[`tiny-gpt-from-scratch`](https://github.com/Abolfazlrwm/Machine-Learning/tree/main/tiny-gpt-from-scratch):
یک موتور **autograd** دست‌ساز، یک ترنسفورمر decoder-only و بهینه‌ساز Adam —
**همه با NumPy خام، بدون PyTorch و بدون JAX** — که روی متن فارسی (همان
corpus دستیار RAG این ریپو، ۹٬۶۶۵ کاراکتر) آموزش دیده است.

* ۲ بلوک · d=96 · ۳ سر توجه · ۲۴۹٬۳۱۶ پارامتر · ۶۵ ثانیه آموزش روی CPU
* هر primitive در تست‌ها در برابر **گرادیان عددی** راستی‌آزمایی شده است

متن تولیدی در سطح «حروف» است (مدل char-level روی دادهٔ خیلی کم است) —
شروع کلمه‌ها، نیم‌فاصله و ریتم نشانه‌گذاری را یاد گرفته ولی روان نیست؛
خودِ این محدودیت بخشی از داستان پروژه است.
