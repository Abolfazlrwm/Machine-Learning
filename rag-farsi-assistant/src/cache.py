"""A small thread-safe TTL + LRU cache (stdlib only).

Used by the RAG pipeline to memoise :meth:`RAGPipeline.answer` results:
same normalised question + retrieval settings within the TTL window ->
same answer, zero retrieval / generation cost.

Design notes
------------
* **TTL + LRU in one structure.** An ``OrderedDict`` keeps insertion order
  (LRU eviction from the front); each entry carries its insertion time from
  an injectable clock, so expiry is deterministic in tests.
* **Stats as first-class citizens.** ``hits`` / ``misses`` / ``expirations``
  / ``evictions`` / ``hit_rate`` are exposed so the serving layer can export
  cache effectiveness alongside retrieval timings.
* An expired entry found on ``get`` is counted under ``expirations`` (not
  double-counted as a miss); ``hit_rate`` uses all three outcomes.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable


class TTLCache:
    def __init__(
        self,
        maxsize: int = 128,
        ttl: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if ttl < 0:
            raise ValueError("ttl must be >= 0")
        self._maxsize = maxsize
        self._ttl = ttl
        self._clock = clock
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0

    # ------------------------------------------------------------- access
    def get(self, key: str) -> tuple[bool, Any]:
        """Return ``(hit, value)``; expired entries count as expirations."""
        now = self._clock()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return False, None
            stored_at, value = item
            if self._ttl and now - stored_at > self._ttl:
                del self._data[key]
                self._expirations += 1
                return False, None
            self._data.move_to_end(key)  # mark as most recently used
            self._hits += 1
            return True, value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            elif len(self._data) >= self._maxsize:
                self._data.popitem(last=False)  # evict the least recently used
                self._evictions += 1
            self._data[key] = (self._clock(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # -------------------------------------------------------------- stats
    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses + self._expirations
            return {
                "hits": self._hits,
                "misses": self._misses,
                "expirations": self._expirations,
                "evictions": self._evictions,
                "size": len(self._data),
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }
