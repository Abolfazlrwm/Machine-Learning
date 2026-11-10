"""Tests for the observability primitives (src/observability.py)."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import MetricsRegistry, Trace  # noqa: E402


# -------------------------------------------------------------- Trace
def test_span_records_wall_time():
    tr = Trace()
    with tr.span("dense"):
        time.sleep(0.01)
    assert tr.timings_ms()["dense"] >= 8


def test_repeated_spans_accumulate_per_stage():
    tr = Trace()
    for _ in range(3):
        with tr.span("bm25"):
            time.sleep(0.005)
    with tr.span("rrf"):
        pass
    assert tr.timings_ms()["bm25"] >= 12
    assert set(tr.timings_ms()) == {"bm25", "rrf"}


def test_notes_flow_into_to_dict():
    tr = Trace()
    tr.note("mode", "hybrid")
    with tr.span("rrf"):
        pass
    dumped = tr.to_dict()
    assert dumped["mode"] == "hybrid"
    assert isinstance(dumped["timings_ms"]["rrf"], int)


def test_span_closes_on_exception():
    tr = Trace()
    with pytest.raises(RuntimeError):
        with tr.span("generate"):
            raise RuntimeError("llm exploded")
    assert "generate" in tr.stages


# ---------------------------------------------------- MetricsRegistry
def test_counter_render_prometheus_format():
    reg = MetricsRegistry()
    reg.describe("http_requests", "HTTP requests processed.")
    reg.inc("http_requests", method="GET", status="200")
    reg.inc("http_requests", method="GET", status="200")
    reg.inc("http_requests", method="POST", status="500")
    text = reg.render()
    assert "# HELP http_requests_total HTTP requests processed." in text
    assert "# TYPE http_requests_total counter" in text
    assert 'http_requests_total{method="GET",status="200"} 2' in text
    assert 'http_requests_total{method="POST",status="500"} 1' in text
    assert text.endswith("\n")


def test_label_values_are_escaped():
    reg = MetricsRegistry()
    reg.inc("weird", path='/x"y\\z')
    assert 'weird_total{path="/x\\"y\\\\z"} 1' in reg.render()


def test_histogram_buckets_are_cumulative_with_inf_and_sum():
    reg = MetricsRegistry(buckets=(0.1, 0.5, 1.0))
    reg.describe("latency", "Latency.")
    for v in (0.05, 0.2, 0.9):
        reg.observe("latency", v)
    text = reg.render()
    assert '# TYPE latency histogram' in text
    assert 'latency_bucket{le="0.1"} 1' in text
    assert 'latency_bucket{le="0.5"} 2' in text   # cumulative
    assert 'latency_bucket{le="1.0"} 3' in text
    assert 'latency_bucket{le="+Inf"} 3' in text
    assert 'latency_sum 1.15' in text
    assert 'latency_count 3' in text


def test_histogram_labels_partition_series():
    reg = MetricsRegistry(buckets=(1.0,))
    reg.observe("dur", 0.5, path="/a")
    reg.observe("dur", 2.0, path="/b")
    text = reg.render()
    # labels are rendered in sorted order (le < path alphabetically)
    assert 'dur_bucket{le="+Inf",path="/a"} 1' in text
    assert 'dur_bucket{le="+Inf",path="/b"} 1' in text


def test_observation_above_last_bucket_lands_only_in_inf():
    reg = MetricsRegistry(buckets=(0.1, 1.0))
    reg.observe("dur", 42.0)
    text = reg.render()
    assert 'dur_bucket{le="0.1"} 0' in text
    assert 'dur_bucket{le="+Inf"} 1' in text


def test_counters_reject_negative_deltas():
    reg = MetricsRegistry()
    with pytest.raises(ValueError):
        reg.inc("c", -1)


def test_concurrent_increments_are_thread_safe():
    reg = MetricsRegistry()
    def worker():
        for _ in range(200):
            reg.inc("events", kind="test")
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 'events_total{kind="test"} 1600' in reg.render()


def test_empty_registry_renders_nothing():
    assert MetricsRegistry().render() == ""
