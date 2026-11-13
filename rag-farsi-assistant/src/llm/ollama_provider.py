"""Ollama provider — local, free, no API key required.

Prerequisites: install Ollama (https://ollama.com), then
    ollama pull llama3.1
    ollama serve
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import requests

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1",
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    def stream(self, system: str, user: str, temperature: float = 0.2) -> Iterator[str]:
        """Yield deltas from Ollama's NDJSON chat stream.

        Each response line is one JSON object: ``{"message": {"content":
        "..."}, "done": false}`` while generating, then a final ``done:
        true`` record that carries only token stats. Every non-empty
        ``message.content`` is yielded as one delta; concatenating the
        yielded deltas reproduces the full reply byte-for-byte.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {"temperature": temperature},
        }
        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            content = json.loads(line).get("message", {}).get("content")
            if content:
                yield content

    def is_available(self) -> bool:
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
