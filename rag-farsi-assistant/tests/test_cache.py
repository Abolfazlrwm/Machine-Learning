"""Tests for the TTL + LRU answer cache (src/cache.py)."""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import TTLCache  # noqa: E402


class FakeClock:
    """Deterministic clock: advance manually (seconds)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_roundtrip_and_miss():
    cache = TTLCache(maxsize=4, ttl=60, clock=FakeClock())
    cache.put("q1", ("answer", []))
    hit, value = cache.get("q1")
    assert hit and value == ("answer", [])
    hit, _ = cache.get("missing")
    assert not hit


def test_ttl_expiry_uses_injected_clock():
    clock = FakeClock()
    cache = TTLCache(maxsize=4, ttl=60, clock=clock)
    cache.put("q1", "v1")
    clock.advance(59)
    assert cache.get("q1")[0] is True
    clock.advance(2)  # now 61s after insertion
    hit, _ = cache.get("q1")
    assert not hit
    assert cache.stats["expirations"] == 1


def test_expired_entries_are_not_double_counted_as_misses():
    clock = FakeClock()
    cache = TTLCache(maxsize=4, ttl=10, clock=clock)
    cache.put("q1", "v1")
    clock.advance(11)
    cache.get("q1")          # expires
    cache.get("never-there")  # true miss
    stats = cache.stats
    assert stats["expirations"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.0


def test_lru_eviction_respects_maxsize():
    cache = TTLCache(maxsize=2, ttl=60, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a" (least recently used)
    assert not cache.get("a")[0]
    assert cache.get("b")[0] and cache.get("c")[0]
    assert cache.stats["evictions"] == 1


def test_get_refreshes_recency():
    cache = TTLCache(maxsize=2, ttl=60, clock=FakeClock())
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")     # "a" becomes most recently used
    cache.put("c", 3)  # evicts "b", keeps "a"
    assert cache.get("a")[0] is True
    assert not cache.get("b")[0]


def test_ttl_zero_never_expires():
    clock = FakeClock()
    cache = TTLCache(maxsize=4, ttl=0, clock=clock)
    cache.put("q1", "v1")
    clock.advance(10_000)
    assert cache.get("q1")[0] is True


def test_overwriting_keeps_insertion_time_of_the_new_value():
    clock = FakeClock()
    cache = TTLCache(maxsize=4, ttl=10, clock=clock)
    cache.put("q1", "old")
    clock.advance(8)
    cache.put("q1", "new")  # fresh timestamp wins
    clock.advance(8)
    hit, value = cache.get("q1")
    assert hit and value == "new"


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        TTLCache(maxsize=0)
    with pytest.raises(ValueError):
        TTLCache(ttl=-1)


def test_concurrent_access_is_thread_safe():
    clock = FakeClock()
    cache = TTLCache(maxsize=64, ttl=60, clock=clock)
    def worker(n: int):
        for i in range(100):
            cache.put(f"k{n}-{i}", i)
            cache.get(f"k{n}-{i}")
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats
    assert stats["size"] == 64          # bounded by maxsize
    assert stats["hits"] + stats["misses"] == 600
