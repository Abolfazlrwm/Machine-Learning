"""Create the configured LLM provider from a config object."""
from __future__ import annotations

from ..config import RAGConfig
from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

_PROVIDERS = {
    OllamaProvider.name: OllamaProvider,
    OpenAIProvider.name: OpenAIProvider,
}


def create_provider(config: RAGConfig, provider: str | None = None) -> LLMProvider:
    """Build a provider by name; falls back to config.provider."""
    name = (provider or config.provider).strip().lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {sorted(_PROVIDERS)}. "
            "For hosted APIs, any OpenAI-compatible endpoint works via 'openai'."
        )
    if name == OllamaProvider.name:
        return OllamaProvider(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=config.request_timeout,
        )
    return OpenAIProvider(
        api_key=config.openai_api_key or None,
        base_url=config.openai_base_url,
        model=config.openai_model,
        timeout=config.request_timeout,
    )
