# One-command developer entry points for the ML portfolio.
# The CI workflows (.github/workflows/) remain the source of truth —
# these targets mirror exactly what CI runs, locally.

PYTHON ?= python3

PROJECTS = rag-farsi-assistant serving tiny-gpt-from-scratch recommender-system ai-text-detector sms-spam stock-movement-ml

LINT_TARGETS = rag-farsi-assistant/src serving/app rag-farsi-assistant/tests serving/tests \
               tiny-gpt-from-scratch/src tiny-gpt-from-scratch/tests \
               recommender-system/src recommender-system/tests \
               ai-text-detector/src ai-text-detector/tests \
               sms-spam/src sms-spam/tests \
               hf-space-tiny-gpt/src scripts/deploy_tiny_gpt_space.py \
               scripts/deploy_serving_space.py

.PHONY: help lint fix test coverage docs-build docs-serve api up down bench space-deploy

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[33m%-14s\033[0m %s\n", $$1, $$2}'

lint:  ## ruff check with the pinned rule set (same list as CI)
	$(PYTHON) -m ruff check $(LINT_TARGETS)

fix:  ## ruff autofix the same paths
	$(PYTHON) -m ruff check --fix $(LINT_TARGETS)

test:  ## run every project's unit tests (movie dataset test auto-skips offline)
	@for p in $(PROJECTS); do \
		echo "==> $$p"; \
		(cd $$p && $(PYTHON) -m pytest tests/ -q) || exit 1; \
	done

coverage:  ## RAG assistant tests with the CI coverage invocation
	cd rag-farsi-assistant && $(PYTHON) -m pytest tests/ -q --cov=src --cov-report=term-missing:skip-covered

docs-build:  ## strict mkdocs build (docs.yml parity)
	$(PYTHON) -m mkdocs build --strict --site-dir site

docs-serve:  ## live-reload docs server on :8000
	$(PYTHON) -m mkdocs serve

api:  ## run the FastAPI serving layer locally
	$(PYTHON) -m uvicorn serving.app.main:create_app --factory --reload --port 8000

up:  ## docker compose: api + ollama + streamlit + prometheus + grafana
	docker compose up -d --build

down:  ## stop the compose stack
	docker compose down

bench:  ## offline latency benchmark for the serving layer
	$(PYTHON) serving/bench.py

space-deploy:  ## redeploy the TinyGPT static Space (needs HF_TOKEN env var)
	$(PYTHON) scripts/deploy_tiny_gpt_space.py

space-deploy-api:  ## deploy the REST API Docker Space (needs HF_TOKEN env var)
	$(PYTHON) scripts/deploy_serving_space.py
