"""Observability primitives: stage tracing + a Prometheus metrics registry.

Two small, dependency-free pieces that make the pipeline measurable:

:class:`Trace`
    Per-request stage timings. The retriever records how long the dense
    search, the BM25 search, the RRF fusion and the re-ranking took; the
    pipeline adds retrieve/generate spans and attaches the totals to the
    answer metadata. Cost: one ``perf_counter`` pair per stage.

:class:`MetricsRegistry`
    Counters and histograms that render the **Prometheus text exposition
    format** (v0.0.4) from scratch — no ``prometheus_client`` dependency,
    roughly 100 lines of stdlib. The serving layer exposes the rendered
    text at ``GET /v1/metrics``.

Both are thread-safe; the registry uses a single lock (contention is
irrelevant at portfolio traffic levels) and stores per-series bucket
counts instead of raw observations, so memory stays bounded.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# ------------------------------------------------------------------- Trace


class Trace:
    """Cumulative per-stage timings plus free-form request notes."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.notes: dict[str, Any] = {}

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time a code block; repeated spans of one name accumulate."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    def timings_ms(self) -> dict[str, int]:
        return {k: int(round(v * 1000)) for k, v in sorted(self.stages.items())}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"timings_ms": self.timings_ms()}
        out.update(self.notes)
        return out


# --------------------------------------------------------------- metrics

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


def _series_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _extend_labels(
    labels: tuple[tuple[str, str], ...], key: str, value: str
) -> tuple[tuple[str, str], ...]:
    return _series_key(dict(labels) | {key: value})


def _format(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(value)


class MetricsRegistry:
    """Counters + histograms rendered as Prometheus text (v0.0.4)."""

    def __init__(
        self,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(buckets))
        self._counters: dict[tuple[str, tuple], float] = {}
        # series -> {"counts": [per-bucket], "sum": float, "count": int}
        self._histograms: dict[tuple[str, tuple], dict[str, Any]] = {}
        self._help: dict[str, str] = {}

    # ------------------------------------------------------------ recording
    def describe(self, name: str, help_text: str) -> None:
        with self._lock:
            self._help[name] = help_text

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        if value < 0:
            raise ValueError("counters can only increase")
        with self._lock:
            key = (name, _series_key(labels))
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        if value < 0:
            raise ValueError("observations must be non-negative")
        with self._lock:
            key = (name, _series_key(labels))
            hist = self._histograms.setdefault(key, {
                "counts": [0] * len(self._buckets), "sum": 0.0, "count": 0,
            })
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    hist["counts"][i] += 1
                    break
            hist["sum"] += value
            hist["count"] += 1

    # ------------------------------------------------------------ rendering
    def render(self) -> str:
        """Serialise to the Prometheus text exposition format (v0.0.4)."""
        lines: list[str] = []
        with self._lock:
            for name in sorted({n for n, _ in self._counters}):
                if name in self._help:
                    lines.append(f"# HELP {name}_total {self._help[name]}")
                lines.append(f"# TYPE {name}_total counter")
                for (hname, labels), value in sorted(self._counters.items()):
                    if hname == name:
                        lines.append(
                            f"{name}_total{_render_labels(labels)} {_format(value)}"
                        )

            for name in sorted({n for n, _ in self._histograms}):
                if name in self._help:
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} histogram")
                for (hname, labels), hist in sorted(self._histograms.items()):
                    if hname != name:
                        continue
                    cumulative = 0
                    for bound, count in zip(self._buckets, hist["counts"]):
                        cumulative += count
                        bucket_labels = _extend_labels(labels, "le", str(bound))
                        lines.append(
                            f"{name}_bucket{_render_labels(bucket_labels)} {cumulative}"
                        )
                    inf_labels = _extend_labels(labels, "le", "+Inf")
                    lines.append(
                        f"{name}_bucket{_render_labels(inf_labels)} {hist['count']}"
                    )
                    lines.append(f"{name}_sum{_render_labels(labels)} {_format(hist['sum'])}")
                    lines.append(f"{name}_count{_render_labels(labels)} {hist['count']}")
        return "\n".join(lines) + ("\n" if lines else "")
