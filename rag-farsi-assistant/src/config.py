"""Central configuration for the Persian RAG assistant.

Every knob lives in one dataclass so the pipeline, CLI and Streamlit app
share the same defaults. Values can be overridden with environment
variables (RAG_*) or programmatically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def _repo_root() -> Path:
    """Project root = folder that contains src/."""
    return Path(__file__).resolve().parent.parent


@dataclass
class RAGConfig:
    # --- corpus / index -------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")
    index_dir: Path = field(default_factory=lambda: _repo_root() / "storage")

    # --- chunking -------------------------------------------------------
    chunk_size: int = 700          # max characters per chunk
    chunk_overlap: int = 150       # characters shared between neighbours

    # --- retrieval ------------------------------------------------------
    top_k: int = 4                 # chunks passed to the LLM
    candidate_multiplier: int = 4  # candidate pool = top_k * multiplier
    mmr_lambda: float = 0.7        # 1.0 = pure relevance, 0.0 = pure diversity
    # dense = embeddings+MMR | sparse = BM25 | hybrid = RRF fusion
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60                # reciprocal rank fusion constant
    bm25_k1: float = 1.5           # BM25 term-frequency saturation
    bm25_b: float = 0.75           # BM25 length normalisation
    reranker: str = "none"         # none | noop | cross-encoder

    # --- embeddings -----------------------------------------------------
    # backend: "sentence-transformers" (real) or "fake" (hash-based, no deps)
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 384  # only used by the "fake" backend

    # --- generation -----------------------------------------------------
    provider: str = "ollama"       # "ollama" | "openai"
    temperature: float = 0.2

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # OpenAI-compatible API (OpenAI, OpenRouter, Groq, ...)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # --- misc -----------------------------------------------------------
    snippet_chars: int = 220       # sources preview length
    request_timeout: int = 120     # seconds for HTTP calls to LLM backends

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Build a config, reading RAG_* environment variables."""
        cfg = cls()
        env_map = {
            "RAG_DATA_DIR": ("data_dir", Path),
            "RAG_INDEX_DIR": ("index_dir", Path),
            "RAG_CHUNK_SIZE": ("chunk_size", int),
            "RAG_CHUNK_OVERLAP": ("chunk_overlap", int),
            "RAG_TOP_K": ("top_k", int),
            "RAG_MMR_LAMBDA": ("mmr_lambda", float),
            "RAG_RETRIEVAL_MODE": ("retrieval_mode", str),
            "RAG_RRF_K": ("rrf_k", int),
            "RAG_BM25_K1": ("bm25_k1", float),
            "RAG_BM25_B": ("bm25_b", float),
            "RAG_RERANKER": ("reranker", str),
            "RAG_EMBEDDING_BACKEND": ("embedding_backend", str),
            "RAG_EMBEDDING_MODEL": ("embedding_model", str),
            "RAG_PROVIDER": ("provider", str),
            "RAG_TEMPERATURE": ("temperature", float),
            "RAG_OLLAMA_BASE_URL": ("ollama_base_url", str),
            "RAG_OLLAMA_MODEL": ("ollama_model", str),
            "RAG_OPENAI_BASE_URL": ("openai_base_url", str),
            "RAG_OPENAI_MODEL": ("openai_model", str),
            "RAG_REQUEST_TIMEOUT": ("request_timeout", int),
        }
        for env_name, (attr, caster) in env_map.items():
            raw = os.environ.get(env_name)
            if raw:
                setattr(cfg, attr, caster(raw))
        cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        cfg.openai_base_url = os.environ.get("OPENAI_BASE_URL", cfg.openai_base_url)
        cfg.openai_model = os.environ.get("OPENAI_MODEL", cfg.openai_model)
        return cfg

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        lines = [f"{f.name} = {getattr(self, f.name)!r}" for f in fields(self)]
        return "\n".join(lines)
