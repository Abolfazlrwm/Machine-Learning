# API Live — explore the serving layer in your browser

This page is the API layer's demo. It combines two things: a real
**Swagger UI** built from the committed OpenAPI snapshot, and **real
captured responses** from the running stack (v2.10.0, extractive
generation mode). To call the endpoints for real, run them locally —
the two commands at the bottom.

## Swagger UI — the full API surface

Schema explorer for every endpoint, request/response model and status
code. *Try-it-out is disabled here because the docs site is static — run
the server locally (below) and the same UI at
`http://localhost:8000/docs` will execute live calls.*

<div id="swagger-ui"></div>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  window.addEventListener('load', function () {
    SwaggerUIBundle({
      url: '../assets/api-openapi.json',
      dom_id: '#swagger-ui',
      supportedSubmitMethods: [],
      docExpansion: 'none',
      displayRequestDuration: true
    });
  });
</script>

## Real captured responses

Recorded from an actual session: `uvicorn serving.app.main:app` with the
real E5 embeddings, the hybrid (E5 + BM25 → RRF) retriever and the
committed SMS artifact. Everything below is verbatim tool output, not a
mock-up.

### `GET /v1/health` — graceful degradation in action

The stock model ships as a training-time artifact (not committed), so the
service reports `unavailable` with a machine-readable contract instead of
crashing — that is the documented behaviour, shown live.

```json
{
  "status": "degraded",
  "version": "2.6.0",
  "services": {
    "rag": "available",
    "sms": "available",
    "stock": "unavailable"
  },
  "uptime_seconds": 19.595
}
```

### `POST /v1/sms/classify` — spam vs ham

```json
{
  "label": "spam",
  "probability_spam": 0.9481,
  "model_version": "tfidf-logreg-gridsearchcv",
  "took_ms": 1
}
```

```json
{
  "label": "ham",
  "probability_spam": 0.0272,
  "model_version": "tfidf-logreg-gridsearchcv",
  "took_ms": 1
}
```

The model is trained on the (English) SMS Spam Collection; an unrelated
Persian message is honestly reported as `ham` — out-of-domain inputs
stay calibrated instead of flipping to spam.

### `POST /v1/rag/ask` — hybrid retrieval with citations

Persian question, hybrid mode, per-stage timings in the response meta:

```json
{
  "question": "بیش‌برازش (overfitting) چیست و چطور جلوی آن را گرفت؟",
  "retrieval_mode": "hybrid",
  "took_ms": 7331,
  "meta": {
    "request_id": "4b28aa38ea2a470984b443bf62397a16",
    "cache": "off",
    "timings_ms": {
      "bm25": 3,
      "dense": 22,
      "generate": 0,
      "retrieve": 24,
      "rrf": 0
    },
    "mixed_script": true
  },
  "sources": [
    {
      "title": "01 machine learning basics",
      "chunk_id": "01_machine_learning_basics::1",
      "score": 0.0273,
      "snippet": "دسته سوم یادگیری تقویتی (Reinforcement Learning) است؛ در این روش یک عامل (Agent) با آزمون و خطا و دریافت پاداش یا جریمه، بهترین رفتار را در یک محیط یاد می‌گیرد؛ بازی‌ها و رباتیک نمونه‌های را…"
    },
    {
      "title": "02 deep learning neural networks",
      "chunk_id": "02_deep_learning_neural_networks::2",
      "score": -0.0107,
      "snippet": "شبکه بازگشتی (RNN) و نسخه بهبودیافته آن LSTM برای داده‌های ترتیبی مثل سری زمانی و متن مناسب‌اند، چون حافظه‌ای از گذشته نگه می‌دارند. اما انقلاب اخیر با معماری ترنسفورمر (Transformer) رخ داد …"
    },
    {
      "title": "04 computer vision",
      "chunk_id": "04_computer_vision::1",
      "score": 0.0946,
      "snippet": "لایه‌های کانولوشن با فیلترهای آموختنی، از لایه‌های اولیه ویژگی‌های ساده مثل لبه و رنگ و از لایه‌های عمیق‌تر اشکال پیچیده و اشیا را می‌آموزند. معماری‌هایی مثل LeNet، AlexNet، VGG، ResNet و Ef…"
    }
  ],
  "answer": "[1] دسته سوم یادگیری تقویتی (Reinforcement Learning) است؛ در این روش یک عامل (Agent) با آزمون و خطا و دریافت پاداش یا جریمه، بهترین رفتار را در یک محیط یاد می‌گیرد؛ بازی‌ها و رباتیک نمونه‌های رایج این دسته‌اند. مهم‌ترین مفهوم در ارزیابی مدل، تقسیم داده به مجموعه آموزش و آزمون است. اگر مدل را روی همان داده‌هایی که دیده ارزیابی کنیم، دچار بیش‌برازش (Overfitting) می‌شویم؛ یعنی مدل مثال‌های آموزشی را حفظ کرده اما توان تعمیم به داده جدید را ندارد. راه‌حل‌های رایج، استفاده از مجموعه اعتبارسنجی، اعتبارسنجی متقاطع (Cross Validation)، منظم‌سازی (Regularization) و ساده نگه‌داشتن مدل است. سنجه‌های ارزیابی نیز بسته به مسئله فرق می‌کنند.\n\n[2] شبکه بازگشتی (RNN) و نسخه بهبودیافته آن LSTM برای داده‌های ترتیبی مثل سری زمانی و متن مناسب‌اند، چون حافظه‌ای از گذشته نگه می‌دارند. اما انقلاب اخیر با معماری ترنسفورمر (Transformer) رخ داد که بر پایه مکانیزم توجه (Attention) بنا شده و برخلاف RNN می‌تواند کل دنباله را به‌صورت موازی پردازش کند. انتخاب معماری به‌تنهایی کافی نیست؛ تنظیم ابرپارامترها مثل نرخ یادگیری، اندازه دسته (Batch Size) و تعداد لایه‌ها نقش بزرگی در کیفیت نهایی دارد. تکنیک‌هایی مثل Dropout، نرمال‌سازی دسته‌ای (Batch Normalization) و افزایش داده (Data Augmentation) نیز برای مقابله با بیش‌برازش در شبکه‌های بزرگ ضروری‌اند.\n\n[3] لایه‌های کانولوشن با فیلترهای آموختنی، از لایه‌های اولیه ویژگی‌های ساده مثل لبه و رنگ و از لایه‌های عمیق‌تر اشکال پیچیده و اشیا را می‌آموزند. معماری‌هایی مثل LeNet، AlexNet، VGG، ResNet و EfficientNet هر نسل، پیشرفت چشمگیری در دقت مجموعه داده ImageNet ایجاد کردند. وظایف اصلی بینایی ماشین عبارت‌اند از: دسته‌بندی تصویر (تعیین موضوع اصلی تصویر)، تشخیص اشیا (یافتن مکان و نوع اشیا با کادر محصورکننده که مدل‌هایی مانند YOLO و Faster R-CNN انجامش می‌دهند)، قطعه‌بندی معنایی (اختصاص برچسب به هر پیکسل با مدل‌هایی مثل U-Net و Mask R-CNN) و تشخیص چهره و ردیابی حرکت."
}
```

### `POST /v1/rag/ask` with `"stream": true` — Server-Sent Events

The SSE stream the API emits for the same question: 5 SSE events: a `sources` event,
incremental `token` deltas, then `done`.

```text
event: sources
data: {"sources": [{"title": "01 machine learning basics", "chunk_id": "01_machine_learning_basics::1", "score": 0.0273, "snippet": "دسته سوم یادگیری تقویتی (Reinforcement Learning) است؛ در این رو …

event: token
data: {"text": "[1] دسته سوم یادگیری تقویتی (Reinforcement Learning) است؛ در این روش یک عامل (Agent) با آزمون و خطا و دریافت پاداش یا جریمه، بهترین رفتار را در یک محیط یاد می‌گیرد؛ بازی‌ها و رباتی …

event: token
data: {"text": "\n\n[2] شبکه بازگشتی (RNN) و نسخه بهبودیافته آن LSTM برای داده‌های ترتیبی مثل سری زمانی و متن مناسب‌اند، چون حافظه‌ای از گذشته نگه می‌دارند. اما انقلاب اخیر با معماری ترنسفورمر (Tr …

event: token
data: {"text": "\n\n[3] لایه‌های کانولوشن با فیلترهای آموختنی، از لایه‌های اولیه ویژگی‌های ساده مثل لبه و رنگ و از لایه‌های عمیق‌تر اشکال پیچیده و اشیا را می‌آموزند. معماری‌هایی مثل LeNet، AlexNet …

event: done
data: {"took_ms": 23, "request_id": "af7e1b73b8f74d3fa6bb75e903b81d13", "cache": "off", "meta": {"cache": "off", "timings_ms": {"bm25": 0, "dense": 22, "generate": 0, "retrieve": 22, "rrf": 0}, "q …

```

### `GET /v1/metrics` — Prometheus scrape endpoint

```text
# HELP http_requests_total HTTP requests processed by the serving layer.
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/openapi.json",status="200"} 1
# HELP rag_cache_requests_total RAG answers by cache outcome: hit, miss or off.
# TYPE rag_cache_requests_total counter
rag_cache_requests_total{result="off"} 3
# HELP http_request_duration_seconds End-to-end request latency in seconds.
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.005",method="GET",path="/openapi.json"} 0
# HELP rag_stage_duration_seconds RAG pipeline stage latency (retrieve, generate) in seconds.
# TYPE rag_stage_duration_seconds histogram
rag_stage_duration_seconds_bucket{le="0.005",stage="generate"} 3
```

## Run it live locally

```bash
pip install -r serving/requirements.txt
ML_REPO_ROOT=$PWD uvicorn serving.app.main:app --port 8000
# Swagger UI: http://localhost:8000/docs
```

Or the full observability stack (API + Prometheus + Grafana):
`make up`. Zero-code paths (both live Spaces and the deployment shapes)
are described in [Deployment](deployment.md).
