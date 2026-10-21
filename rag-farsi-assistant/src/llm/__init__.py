"""LLM provider subpackage: swap Ollama <-> OpenAI-compatible with one line."""
from .base import LLMProvider
from .factory import create_provider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

__all__ = ["LLMProvider", "create_provider", "OllamaProvider", "OpenAIProvider"]
