"""Serving-layer metrics singleton.

The registry implementation is the zero-dependency one from
``rag-farsi-assistant/src/observability.py``. It is loaded **by file path**
(not by putting the RAG project on ``sys.path``) so importing this module
never changes how other modules resolve the ambiguous ``src`` package name
— see ``services.py`` for why that matters.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_OBSERVABILITY_FILE = (
    Path(__file__).resolve().parents[2] / "rag-farsi-assistant" / "src" / "observability.py"
)


def _load_observability():
    spec = importlib.util.spec_from_file_location("rag_observability", _OBSERVABILITY_FILE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load observability from {_OBSERVABILITY_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rag_observability"] = module
    spec.loader.exec_module(module)
    return module


_observability = _load_observability()
MetricsRegistry = _observability.MetricsRegistry

metrics = MetricsRegistry()
metrics.describe("http_requests", "HTTP requests processed by the serving layer.")
metrics.describe(
    "http_request_duration_seconds", "End-to-end request latency in seconds."
)
metrics.describe(
    "rag_cache_requests",
    "RAG answers by cache outcome: hit, miss or off.",
)
metrics.describe(
    "rag_stage_duration_seconds",
    "RAG pipeline stage latency (retrieve, generate) in seconds.",
)
