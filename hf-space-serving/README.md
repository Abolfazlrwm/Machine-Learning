---
title: ML Portfolio REST API
emoji: 🔌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ML Portfolio — live REST API

One FastAPI serving layer over the portfolio's model families, deployed as
a Hugging Face Docker Space. Open **`/docs`** for the interactive
Swagger UI and call the endpoints right from the browser:

- `GET /v1/health` — which services are up (rag / sms / stock)
- `POST /v1/sms/predict` — spam vs ham on real messages
- `POST /v1/rag/ask` — Persian RAG with numbered citations
- `GET /v1/metrics` — Prometheus scrape endpoint

Source of truth: [Abolfazlrwm/Machine-Learning](https://github.com/Abolfazlrwm/Machine-Learning)
(`serving/`), including the test suite (40 tests), SSE streaming on
`/v1/rag/ask?stream=true` and the Dockerfile in `hf-space-serving/`.
