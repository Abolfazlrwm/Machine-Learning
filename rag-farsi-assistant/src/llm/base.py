"""Abstract LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    """A thin wrapper around any chat-capable LLM backend."""

    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Return the assistant reply for one system+user message pair."""

    def stream(self, system: str, user: str, temperature: float = 0.2) -> Iterator[str]:
        """Yield the reply incrementally as text deltas.

        The default implementation buffers :meth:`generate` and yields the
        whole reply as one delta, so *every* provider is stream-capable out
        of the box and callers never need an ``isinstance`` check. Backends
        with a real streaming protocol (OpenAI SSE, Ollama NDJSON) override
        this and yield genuine first-token deltas instead.
        """
        yield self.generate(system, user, temperature)

    def is_available(self) -> bool:
        """Cheap readiness check (used by the UI for a friendly error)."""
        return True
