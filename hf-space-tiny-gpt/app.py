"""Live demo of TinyGPT-from-scratch — a Persian char-level GPT with zero
ML frameworks.

The whole stack is vendored from the repo's `tiny-gpt-from-scratch/src`
(autograd engine + transformer + tokenizer) and runs the committed 2 MB
float64 checkpoint (`model/params.npz`, 249k params) straight on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
from src.model import TinyGPT
from src.tokenizer import CharTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "model"
REPO_URL = "https://github.com/Abolfazlrwm/Machine-Learning/tree/main/tiny-gpt-from-scratch"


def load_model() -> tuple[TinyGPT, CharTokenizer, dict]:
    log = json.loads((MODEL_DIR / "train_log.json").read_text(encoding="utf-8"))
    config = log["config"]
    tok = CharTokenizer.from_json(MODEL_DIR / "tokenizer.json")
    state = {k: np.asarray(v) for k, v in np.load(MODEL_DIR / "params.npz").items()}
    model = TinyGPT(config["vocab"], dim=config["dim"], n_layers=config["n_layers"],
                    n_heads=config["n_heads"], max_t=config["max_t"])
    model.load_state_dict(state)
    return model, tok, log


MODEL, TOKENIZER, LOG = load_model()


def generate(prompt: str, temperature: float, top_k: int,
             n_chars: int, seed: int) -> tuple[str, str]:
    if not prompt.strip():
        return "", "یک پرامپت فارسی بنویسید."
    rng = np.random.default_rng(seed)
    prompt_ids = TOKENIZER.encode(prompt)
    if not prompt_ids:
        return prompt, "هیچ نویسه‌ای از پرامپت در واژگان ۱۰۰ نویسه‌ای مدل نبود."
    ids = MODEL.sample(prompt_ids, n_new=int(n_chars), temperature=float(temperature),
                       top_k=int(top_k) if top_k > 0 else None, rng=rng)
    text = TOKENIZER.decode(ids)
    info = (f"vocabulary {LOG['config']['vocab']} chars · "
            f"{LOG['config']['n_params']:,} params · trained {LOG['elapsed_sec']} s "
            f"on CPU · final train loss {LOG['final_train_loss']:.2f} / val {LOG['final_val_loss']:.2f}")
    return text, info


with gr.Blocks(title="TinyGPT فارسی — GPT از صفر") as demo:
    gr.Markdown(
        f"""
# 🧠 تولید متن فارسی با GPT ساخته‌شده از صفر

موتور autograd + ترنسفورمر + Adam، **همه با NumPy خام** — بدون PyTorch.
کد و آموزش: [`tiny-gpt-from-scratch`]({REPO_URL}) در
[پرتفولیوی ML](https://github.com/Abolfazlrwm/Machine-Learning).
متن را بنویسید و ادامهٔ آن را از یک مدل char-level ۲۴۹هزار-پارامتری بگیرید.
"""
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="پرامپت فارسی", value="یادگیری ماشین",
                                placeholder="مثلاً: یادگیری ماشین …")
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05,
                                        label="Temperature (کم‌تر = محافظه‌کارتر)")
                top_k = gr.Slider(1, 50, value=12, step=1,
                                  label="Top-k (فقط k نویسهٔ محتمل‌تر)")
            n_chars = gr.Slider(50, 400, value=220, step=10,
                                label="طول ادامه (نویسه)")
            seed = gr.Number(value=7, precision=0, label="Seed (برای تکرار‌پذیری)")
            btn = gr.Button("تولید کن", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="خروجی مدل", lines=10)
            stats = gr.Markdown()

    btn.click(generate, [prompt, temperature, top_k, n_chars, seed],
              [output, stats])

    gr.Examples(
        examples=[["یادگیری ماشین", 0.8, 12, 220, 7],
                  ["شبکهٔ عصبی", 0.9, 20, 300, 21],
                  ["بیش‌برازش چیست", 0.7, 8, 200, 3]],
        inputs=[prompt, temperature, top_k, n_chars, seed],
    )

    gr.Markdown(
        """
**این مدل چه بلدی؟** حروف، شروع واژه‌ها، نیم‌فاصله و نشانه‌گذاری فارسی را
یاد گرفته؛ روانی نویسه‌ای نیازمند دادهٔ چند مرتبهٔ بزرگ‌تر است — loss آموزش
۱٫۲۲ در برابر ۲٫۷۴ اعتبارسنجی یعنی مدل روی ۹٬۶۶۵ کاراکتر حفظ‌کردن را شروع
کرده. این محدودیت صادقانه، بخشی از نتیجه است.
"""
    )

if __name__ == "__main__":
    demo.launch()
