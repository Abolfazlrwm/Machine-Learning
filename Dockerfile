# syntax=docker/dockerfile:1

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
    RAG_EMBEDDING_BACKEND=sentence-transformers

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Serving layer
COPY serving/__init__.py serving/__init__.py
COPY serving/app ./serving/app

# RAG assistant (source + corpus; the FAISS/BM25 index builds on first start)
COPY rag-farsi-assistant/src ./rag-farsi-assistant/src
COPY rag-farsi-assistant/data ./rag-farsi-assistant/data

# Stock + SMS sources (artifacts are expected as volumes, see compose file)
COPY stock-movement-ml/src ./stock-movement-ml/src
COPY sms-spam ./sms-spam

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4).status < 500 else 1)"

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
