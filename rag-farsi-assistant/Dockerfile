# Streamlit chat UI for the Persian RAG assistant.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data
COPY app.py main.py evaluate.py ./

# Streamlit writes ~/.streamlit config state; a named home directory for the
# runtime user keeps that working without root.
RUN useradd --system --uid 1000 --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
