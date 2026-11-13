"""Streaming contract of the LLM layer (src/llm/base.py + providers).

``LLMProvider.stream`` must be available on *every* provider: the default
buffers ``generate()`` into one delta, and real backends (OpenAI SSE,
Ollama NDJSON) override it with genuine incremental deltas. Concatenating
the yielded deltas always reproduces the ``generate()`` reply — that is
the invariant the pipeline cache relies on.
"""
from __future__ import annotations

import json

import pytest
import requests
from src.llm.base import LLMProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAIProvider

SYSTEM = "sys"
USER = "usr"


class BufferingProvider(LLMProvider):
    """Provider that only implements ``generate`` — inherits the default."""

    name = "buffering"

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        return "پاسخ کامل و یک‌تکه."


class StreamingProvider(LLMProvider):
    """Provider with a real multi-delta stream override."""

    name = "streaming"

    def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        return "دلتا یکدلتا دو"

    def stream(self, system: str, user: str, temperature: float = 0.2):
        yield "دلتا "
        yield "یک"
        yield "دلتا دو"


class FakeStreamResponse:
    """Minimal stand-in for ``requests`` streaming responses."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self, decode_unicode: bool = True):
        return iter(self._lines)


class FailingResponse(FakeStreamResponse):
    def raise_for_status(self) -> None:
        raise requests.HTTPError("500 Server Error")


def test_default_stream_buffers_generate_into_one_delta():
    provider = BufferingProvider()
    deltas = list(provider.stream(SYSTEM, USER))
    assert deltas == ["پاسخ کامل و یک‌تکه."]
    assert "".join(deltas) == provider.generate(SYSTEM, USER)


def test_stream_override_yields_real_deltas():
    provider = StreamingProvider()
    deltas = list(provider.stream(SYSTEM, USER))
    assert deltas == ["دلتا ", "یک", "دلتا دو"]
    assert "".join(deltas) == provider.generate(SYSTEM, USER)


def test_openai_stream_parses_sse_deltas(monkeypatch):
    sse_lines = [
        'data: {"choices": [{"delta": {"role": "assistant"}}]}',
        'data: {"choices": [{"delta": {"content": "بیش‌"}}]}',
        ":",  # SSE keep-alive comment
        'data: {"choices": [{"delta": {"content": "برازش"}}]}',
        'data: {"choices": [{"delta": {"content": " چیست"}}]}',
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "IGNORED"}}]}',
    ]
    seen: dict = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["payload"] = kwargs["json"]
        return FakeStreamResponse(sse_lines)

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OpenAIProvider(api_key="test-key", base_url="http://fake/v1")

    deltas = list(provider.stream(SYSTEM, USER))
    assert deltas == ["بیش‌", "برازش", " چیست"]
    assert "".join(deltas) == "بیش‌برازش چیست"
    assert seen["url"] == "http://fake/v1/chat/completions"
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["messages"][0]["role"] == "system"


def test_openai_stream_surfaces_http_errors(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **kw: FailingResponse([]))
    provider = OpenAIProvider(api_key="test-key", base_url="http://fake/v1")
    with pytest.raises(requests.HTTPError):
        list(provider.stream(SYSTEM, USER))


def test_ollama_stream_parses_ndjson_deltas(monkeypatch):
    ndjson_lines = [
        json.dumps({"message": {"role": "assistant", "content": "پاسخ"}, "done": False}),
        json.dumps({"message": {"content": " آزمایشی"}, "done": False}),
        "",  # blank line between records
        json.dumps({"done": True, "total_duration": 123}),
    ]

    def fake_post(url, **kwargs):
        assert kwargs["json"]["stream"] is True
        return FakeStreamResponse(ndjson_lines)

    monkeypatch.setattr(requests, "post", fake_post)
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")

    deltas = list(provider.stream(SYSTEM, USER))
    assert deltas == ["پاسخ", " آزمایشی"]
    assert "".join(deltas) == "پاسخ آزمایشی"


def test_ollama_stream_surfaces_http_errors(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda url, **kw: FailingResponse([]))
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.1")
    with pytest.raises(requests.HTTPError):
        list(provider.stream(SYSTEM, USER))
