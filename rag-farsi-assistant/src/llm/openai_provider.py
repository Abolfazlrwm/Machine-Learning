"""OpenAI-compatible chat provider.

Works with OpenAI itself *and* any OpenAI-compatible endpoint
(OpenRouter, Groq, Together, local llama.cpp server, ...):

    export OPENAI_API_KEY=...            # required
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional
    export OPENAI_MODEL=meta-llama/llama-3.1-8b-instruct  # optional
"""
from __future__ import annotations

import os

import requests

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "OpenAI provider needs an API key. Set OPENAI_API_KEY "
                "(or pass api_key=...) — works with OpenAI/OpenRouter/Groq keys."
            )

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
