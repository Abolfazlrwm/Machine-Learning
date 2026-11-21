# syntax=docker/dockerfile:1
# HF Spaces (Docker SDK) packaging of the ML portfolio serving layer.
# Same stack as the repo-root Dockerfile, adapted for Spaces:
#   * port 7860 (Spaces' required public port)
#   * the RAG index + E5 model cache are pre-baked at BUILD time, so the
#     Space starts fast instead of downloading the model on the first ask
# NOTE: COPY paths must mirror scripts/deploy_serving_space.py staging.

########################################################################
# Builder — compile the full serving stack into a standalone venv
########################################################################
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY serving/requirements-docker.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements-docker.txt

########################################################################
# Runtime — slim image, non-root, one command to serve everything
########################################################################
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ML_REPO_ROOT=/app \
    RAG_EMBEDDING_BACKEND=sentence-transformers \
    HF_HOME=/app/.hf-cache

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Serving layer
COPY serving/__init__.py serving/__init__.py
COPY serving/app ./serving/app

# RAG assistant (source + corpus; index pre-baked below)
COPY rag-farsi-assistant/src ./rag-farsi-assistant/src
COPY rag-farsi-assistant/data ./rag-farsi-assistant/data

# Stock + SMS sources; SMS ships its committed artifact
COPY stock-movement-ml/src ./stock-movement-ml/src
COPY sms-spam ./sms-spam

RUN useradd --create-home appuser \
 && mkdir -p /app/.hf-cache \
 && chown -R appuser:appuser /app
USER appuser

# Pre-bake the RAG index (E5 model downloaded once into HF_HOME) so the
# Space is warm the moment it reports "Running".
RUN python - <<'PY'
import sys
sys.path.insert(0, "/app/rag-farsi-assistant")
from src.config import RAGConfig
from src.pipeline import RAGPipeline

pipeline = RAGPipeline(RAGConfig())
print("pre-baking index:", pipeline.build_index(), flush=True)
PY

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/v1/health', timeout=4).status < 500 else 1)"

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
