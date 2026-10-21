"""Abstract LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """A thin wrapper around any chat-capable LLM backend."""

    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Return the assistant reply for one system+user message pair."""

    def is_available(self) -> bool:
        """Cheap readiness check (used by the UI for a friendly error)."""
        return True
