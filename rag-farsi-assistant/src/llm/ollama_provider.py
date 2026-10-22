"""Ollama provider — local, free, no API key required.

Prerequisites: install Ollama (https://ollama.com), then
    ollama pull llama3.1
    ollama serve
"""
from __future__ import annotations

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
