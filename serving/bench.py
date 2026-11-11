"""Offline latency benchmark for the serving API (single process, no network).

Why
---
CI gives you correctness; this gives you a *shape* of performance you can
put next to the correctness numbers. It drives the real FastAPI app over
httpx's ASGI transport with the RAG pipeline running on the deterministic
fake embedder and extractive answers — so the measured path is everything
except model inference: HTTP middleware, auth, query folding/expansion,
BM25 + vector search, RRF fusion, caching and JSON/SSE serialisation.

Scenarios
---------
health          GET /v1/health — the floor of the stack (no model work).
cache-off       POST /v1/rag/ask with RAG_CACHE_TTL=0 (default) — every
                request runs the full retrieval pipeline.
cache-exact     TTL cache on — repeats hit; paraphrases miss.
cache-semantic  Semantic cache on — repeats *and* paraphrases hit.

The workload mixes 80% repeats of a question pool with 20% paraphrases
(word-order swaps / rewordings of the same pool entries). Cache-off sees
them all as misses; the exact cache only catches the repeats; the
semantic cache catches both. Observed hit rates are parsed from the
responses' meta block — the report never assumes them.

Usage
-----
    python serving/bench.py                    # all scenarios, N=200, C=10
    python serving/bench.py -n 500 -c 20       # more load
    python serving/bench.py --output report.md # write markdown next to stdout

Interpretation
--------------
Single process, in-process transport: no network, no uvicorn worker
overhead, GIL shared across concurrency. Numbers are indicative of the
relative cost of each layer, not of a deployed multi-worker deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from serving.app.main import create_app  # noqa: E402
from serving.app.services import ServiceRegistry  # noqa: E402

QUESTION_POOL = [
    "بیش‌برازش چیست",
    "یادگیری با نظارت چیست",
    "خوشه‌بندی چیست",
    "شبکه عصبی چگونه کار می‌کند",
    "مکانیزم توجه چیست",
    "پس‌انتشار خطا چیست",
    "بهینه‌ساز Adam چه کاری می‌کند",
    "رگرسیون خطی چیست",
]

PARAPHRASE_PREFIX = ["لطفا توضیح بده", "به زبان ساده", "خلاصه کن"]


def paraphrase(question: str, i: int) -> str:
    """Cheap deterministic rewording: prefix + word-order flip."""
    words = question.split()
    flipped = " ".join(reversed(words))
    return f"{PARAPHRASE_PREFIX[i % len(PARAPHRASE_PREFIX)]} {flipped}"


@dataclass
class ScenarioResult:
    name: str
    description: str
    n: int
    concurrency: int
    latencies_ms: list[float]
    wall_s: float
    cache_counts: dict[str, int]

    @property
    def rps(self) -> float:
        return self.n / self.wall_s

    def pct(self, p: float) -> float:
        data = sorted(self.latencies_ms)
        idx = min(len(data) - 1, round(p / 100 * (len(data) - 1)))
        return data[idx]

    def hit_rate(self) -> float:
        hits = self.cache_counts.get("hit", 0)
        total = sum(self.cache_counts.values()) or 1
        return 100.0 * hits / total


def _set_env(scenario: str, index_dir: Path) -> None:
    os.environ["RAG_EMBEDDING_BACKEND"] = "fake"
    # No key -> the OpenAI provider refuses to build -> the registry falls
    # back to extractive answers. Fully offline, fully deterministic.
    os.environ["RAG_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["RAG_INDEX_DIR"] = str(index_dir)
    os.environ.pop("RAG_RERANKER", None)
    if scenario == "cache-off":
        os.environ["RAG_CACHE_TTL"] = "0"
        os.environ["RAG_SEMANTIC_CACHE"] = "0"
    elif scenario == "cache-exact":
        os.environ["RAG_CACHE_TTL"] = "600"
        os.environ["RAG_SEMANTIC_CACHE"] = "0"
    elif scenario == "cache-semantic":
        os.environ["RAG_CACHE_TTL"] = "600"
        os.environ["RAG_SEMANTIC_CACHE"] = "1"
        os.environ["RAG_SEMANTIC_CACHE_THRESHOLD"] = "0.5"


def build_workload(n: int) -> list[str]:
    """80% pool repeats + 20% paraphrases of the same entries."""
    workload = []
    for i in range(n):
        base = QUESTION_POOL[i % len(QUESTION_POOL)]
        if i % 5 == 4:
            workload.append(paraphrase(base, i))
        else:
            workload.append(base)
    return workload


async def run_scenario(
    scenario: str, n: int, concurrency: int, index_dir: Path
) -> ScenarioResult:
    _set_env(scenario, index_dir)
    app = create_app(registry=ServiceRegistry(root=REPO_ROOT))

    descriptions = {
        "health": "GET /v1/health — stack floor, no model work",
        "cache-off": "POST /v1/rag/ask — cache disabled, full pipeline each time",
        "cache-exact": "POST /v1/rag/ask — exact TTL cache (repeats hit)",
        "cache-semantic": "POST /v1/rag/ask — semantic cache (repeats + paraphrases hit)",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://bench", timeout=60.0
    ) as client:
        # Warm-up: builds the index once (excluded from all timings).
        resp = await client.post("/v1/rag/ask", json={"question": QUESTION_POOL[0]})
        resp.raise_for_status()

        workload = build_workload(n)
        latencies: list[float] = []
        cache_counts = {"hit": 0, "miss": 0, "off": 0}
        sem = asyncio.Semaphore(concurrency)

        async def one(question: str) -> None:
            async with sem:
                start = time.perf_counter()
                if scenario == "health":
                    resp = await client.get("/v1/health")
                else:
                    resp = await client.post("/v1/rag/ask", json={"question": question})
                elapsed = (time.perf_counter() - start) * 1000.0
                resp.raise_for_status()
                if scenario != "health":
                    state = (resp.json().get("meta") or {}).get("cache", "off")
                    cache_counts[state] = cache_counts.get(state, 0) + 1
                latencies.append(elapsed)

        wall_start = time.perf_counter()
        await asyncio.gather(*(one(q) for q in workload))
        wall = time.perf_counter() - wall_start

    return ScenarioResult(
        name=scenario,
        description=descriptions[scenario],
        n=n,
        concurrency=concurrency,
        latencies_ms=latencies,
        wall_s=wall,
        cache_counts=cache_counts,
    )


def render_report(results: list[ScenarioResult], n: int, c: int) -> str:
    lines = [
        "# Serving API — offline latency benchmark",
        "",
        "Single-process ASGI benchmark (`httpx.ASGITransport`) of the real FastAPI",
        "app: RAG on the deterministic fake embedder, extractive answers, no",
        "network and no model inference. Measures middleware + query intelligence",
        "+ BM25/vector retrieval + RRF + caching + serialisation.",
        "",
        f"- workload: `{n}` requests at concurrency `{c}` — 80% repeats of an",
        "  8-question Persian pool, 20% paraphrases (prefix + word-order swap)",
        "- machine: benchmark runner inside CI-equivalent container, one process",
        "- hit rate: parsed from the responses' `meta.cache`, never assumed",
        "",
        "| scenario | description | cache hit | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | max ms | req/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.name}` | {r.description} | {r.hit_rate():.0f}% "
            f"| {r.pct(50):.1f} | {r.pct(90):.1f} | {r.pct(95):.1f} "
            f"| {r.pct(99):.1f} | {statistics.mean(r.latencies_ms):.1f} "
            f"| {max(r.latencies_ms):.1f} | {r.rps:.0f} |"
        )
    lines += [
        "",
        "## What the numbers say",
        "",
        "- `cache-off` is the honest cost of one full hybrid retrieval per request;",
        "- `cache-exact` cuts latency for repeated questions but every paraphrase",
        "  still pays the full price — hit rate stays at the repeat share;",
        "- `cache-semantic` absorbs the paraphrases too (the hit rate climbs to the",
        "  repeat + paraphrase share) while adding one query embedding per miss.",
        "",
        "Single process and GIL-bound: absolute req/s understates a deployed",
        "multi-worker setup — read the columns *relative* to `cache-off`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=200, help="requests per scenario")
    parser.add_argument("-c", type=int, default=10, help="concurrency")
    parser.add_argument("--output", type=Path, default=None, help="write markdown here")
    args = parser.parse_args()

    index_dir = REPO_ROOT / "rag-farsi-assistant" / "storage-bench"
    results = []
    try:
        for scenario in ("health", "cache-off", "cache-exact", "cache-semantic"):
            print(f"running {scenario} ...", file=sys.stderr)
            results.append(asyncio.run(run_scenario(scenario, args.n, args.c, index_dir)))
    finally:
        if index_dir.exists():
            import shutil

            shutil.rmtree(index_dir, ignore_errors=True)

    report = render_report(results, args.n, args.c)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"written: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
