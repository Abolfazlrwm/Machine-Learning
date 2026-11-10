"""Lazy service registry binding the three ML projects to the API.

Design goals
------------
* **Fail loud, degrade clean.** Every service reports whether its local
  artifacts (model / index / dataset) exist. Missing artifacts surface as a
  503 with a actionable hint instead of a stack trace.
* **Lazy singletons.** Heavy objects (RAG pipeline, sklearn models) are built
  on first use behind a lock, so importing this module stays cheap and CI
  never needs the heavy stack.
* **No package-name collisions.** ``rag-farsi-assistant/src`` and
  ``stock-movement-ml/src`` are both packages named ``src`` — the RAG one is
  imported normally (once), stock modules are loaded by file path via
  ``importlib`` to avoid the clash.

Environment variables:
    ML_REPO_ROOT            repo root (default: auto-detected)
    SERVING_API_KEY         if set, /v1 endpoints require X-API-Key
    RAG_*                   all knobs of the RAG assistant (see its .env.example)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .exceptions import NotFound, ServiceUnavailable, UpstreamError


def repo_root() -> Path:
    """Repo root: ``ML_REPO_ROOT`` or two levels above this file."""
    env = os.environ.get("ML_REPO_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------- helpers
def _module_from_file(name: str, path: Path):
    """Import a single .py file under a unique module name (no package clash)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ServiceStatus:
    available: bool
    detail: str


class BaseLazyService:
    """Common lazy-initialisation + status plumbing."""

    name = "base"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._built_at: float | None = None

    def available(self) -> ServiceStatus:
        """Cheap filesystem/stdlib probe — no heavy imports, no locks."""
        raise NotImplementedError

    def _build(self) -> None:
        """Construct the heavyweight backing object (called under lock)."""
        raise NotImplementedError

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._build()
            self._ready = True
            self._built_at = time.monotonic()


# --------------------------------------------------------------------- RAG
class RAGService(BaseLazyService):
    """Persian RAG assistant: retrieval + (optional) grounded generation."""

    name = "rag"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root
        self._project = root / "rag-farsi-assistant"
        self._pipeline = None
        self._retrieval_mode = "unknown"

    def available(self) -> ServiceStatus:
        if not (self._project / "src" / "config.py").exists():
            return ServiceStatus(False, "rag-farsi-assistant source not found")
        has_index = (self._project / "storage" / "meta.json").exists()
        has_corpus = any((self._project / "data").glob("*.txt"))
        if not (has_index or has_corpus):
            return ServiceStatus(False, "no index and no corpus documents")
        backend = os.environ.get("RAG_EMBEDDING_BACKEND", "sentence-transformers")
        if backend == "sentence-transformers" and importlib.util.find_spec(
            "sentence_transformers"
        ) is None:
            return ServiceStatus(
                False,
                "embedding backend 'sentence-transformers' not installed "
                "(set RAG_EMBEDDING_BACKEND=fake for a smoke deployment)",
            )
        return ServiceStatus(True, f"index={has_index}, corpus={has_corpus}")

    def _build(self) -> None:
        sys.path.insert(0, str(self._project))
        try:
            from src.config import RAGConfig  # noqa: E402  (path set above)
            from src.llm.factory import create_provider  # noqa: E402
            from src.pipeline import RAGPipeline  # noqa: E402
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise ServiceUnavailable(
                f"cannot import the RAG assistant: {exc}",
                hint="deploy with the full image (serving/requirements-docker.txt)",
            ) from exc

        try:
            config = RAGConfig.from_env()
            config.ensure_dirs()
            self._retrieval_mode = config.retrieval_mode
            provider = None
            try:
                provider = create_provider(config)
            except Exception:  # noqa: BLE001 - generation is optional
                provider = None  # extractive fallback keeps the API useful
            self._pipeline = RAGPipeline(config, llm=provider)
            self._pipeline.ensure_index()
        except ImportError as exc:
            # Heavy backend missing (e.g. sentence-transformers not installed)
            # -> the clean-degradation contract is a 503, not a stack trace.
            raise ServiceUnavailable(
                f"RAG assistant is missing a required backend: {exc}",
                hint="install serving/requirements-docker.txt or set "
                     "RAG_EMBEDDING_BACKEND=fake for a smoke deployment",
            ) from exc

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Return {question, answer, sources, retrieval_mode} — raises on failure."""
        self.ensure_ready()
        try:
            result = self._pipeline.answer(question, top_k=top_k)
        except ServiceUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - LLM/network failures land here
            raise UpstreamError(
                f"generation failed: {exc}",
                hint="check the LLM provider (RAG_PROVIDER / Ollama / OpenAI key)",
            ) from exc
        return {
            "question": result.question,
            "answer": result.answer,
            "sources": [vars(s) for s in result.sources],
            "retrieval_mode": self._retrieval_mode,
            "meta": dict(getattr(result, "meta", {}) or {}),
        }


# --------------------------------------------------------------------- SMS
class SmsService(BaseLazyService):
    """TF-IDF + LogisticRegression spam classifier (sms-spam project)."""

    name = "sms"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._model_path = root / "sms-spam" / "spam_classifier.pkl"
        self._model = None
        self._version = "unavailable"

    def available(self) -> ServiceStatus:
        if self._model_path.exists():
            return ServiceStatus(True, str(self._model_path.name))
        return ServiceStatus(
            False,
            "sms-spam/spam_classifier.pkl not found — train it with "
            "`python spam_classifier.py --train spam.csv`",
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        """Mirror of sms-spam clean_text (kept import-light for the API)."""
        import re
        import string as _string

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", _string.punctuation))
        words = text.split()
        try:  # NLTK is optional at serve time; fall back to stopword-less pass
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            stop = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            words = [
                lemmatizer.lemmatize(w) for w in words if w not in stop
            ]
        except LookupError:  # corpora not downloaded
            pass
        except ImportError:
            pass
        return " ".join(words)

    def _build(self) -> None:
        if not self._model_path.exists():
            raise ServiceUnavailable(
                "spam classifier artifact missing",
                hint="run `python spam_classifier.py --train spam.csv` in sms-spam/",
            )
        import joblib  # lazy: only needed when the service is actually used

        self._model = joblib.load(self._model_path)
        self._version = "tfidf-logreg-gridsearchcv"

    def classify(self, text: str) -> dict:
        self.ensure_ready()
        cleaned = self._clean_text(text)
        proba = float(self._model.predict_proba([cleaned])[0][1])
        return {
            "label": "spam" if proba >= 0.5 else "ham",
            "probability_spam": round(proba, 4),
            "model_version": self._version,
        }


# ------------------------------------------------------------------- stock
class StockService(BaseLazyService):
    """Serve persisted walk-forward models for movement signals."""

    name = "stock"

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._project = root / "stock-movement-ml"
        self._features_mod = None

    # -- probes ----------------------------------------------------------
    def _tickers_with_artifacts(self) -> list[str]:
        data = self._project / "data"
        models = self._project / "models"
        if not (data.exists() and models.exists()):
            return []
        tickers = {p.name.split("_")[0].upper() for p in data.glob("*_*.csv")}
        modelled = {p.name.split("_")[0].upper() for p in models.glob("*_*.joblib")}
        return sorted(tickers & modelled)

    def available(self) -> ServiceStatus:
        tickers = self._tickers_with_artifacts()
        if tickers:
            return ServiceStatus(True, f"tickers={','.join(tickers)}")
        return ServiceStatus(
            False,
            "no cached data + persisted models — run `python main.py --ticker AAPL`",
        )

    def _load_features_module(self):
        if self._features_mod is None:
            features_file = self._project / "src" / "features.py"
            if not features_file.exists():
                raise ServiceUnavailable("stock-movement-ml source not found")
            self._features_mod = _module_from_file(
                "stock_features_serving", features_file
            )
        return self._features_mod

    def _build(self) -> None:
        # Pre-import the (pandas-only) feature module; models load per request.
        self._load_features_module()

    # -- inference -------------------------------------------------------
    def _latest_model_path(self, ticker: str) -> Path:
        models_dir = self._project / "models"
        best_name = None
        report = self._project / "reports" / "backtest_metrics.json"
        if report.exists():
            import json

            best_name = json.loads(report.read_text()).get("model")
        candidates = []
        if best_name:
            candidates.append(models_dir / f"{ticker}_{best_name}.joblib")
        candidates += [
            models_dir / f"{ticker}_Random_Forest.joblib",
            models_dir / f"{ticker}_XGBoost.joblib",
            models_dir / f"{ticker}_Logistic_Regression.joblib",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise NotFound(
            f"no persisted model for ticker {ticker!r}",
            hint="run the stock pipeline first: python main.py --ticker AAPL",
        )

    def signal(self, ticker: str) -> dict:
        # Guards first: the 404 degradation path must work in the lean
        # runtime where joblib/pandas are intentionally absent (CI, slim
        # images) — heavy imports only happen once artifacts exist.
        ticker = ticker.upper()
        data_dir = self._project / "data"
        csvs = sorted(
            data_dir.glob(f"{ticker}_*.csv"), key=lambda p: p.stat().st_mtime
        )
        if not csvs:
            raise NotFound(
                f"no cached OHLCV data for {ticker!r}",
                hint="run the stock pipeline first: python main.py --ticker AAPL",
            )
        model_path = self._latest_model_path(ticker)
        self.ensure_ready()

        import joblib  # lazy heavy imports
        import pandas as pd

        raw = pd.read_csv(csvs[-1], index_col=0, parse_dates=True)
        features = self._load_features_module().calculate_technical_indicators(raw)
        feature_cols = self._load_features_module().FEATURE_COLUMNS
        usable = features.dropna(subset=feature_cols)
        if usable.empty:
            raise ServiceUnavailable(
                "not enough history to compute technical indicators",
                hint="fetch a longer period (main.py --start 2018-01-01)",
            )
        last_row, as_of = usable.iloc[-1], usable.index[-1]
        model = joblib.load(model_path)
        prob_up = float(model.predict_proba(last_row[feature_cols].to_frame().T)[0][1])
        return {
            "ticker": ticker,
            "signal": "up" if prob_up >= 0.5 else "down",
            "probability_up": round(prob_up, 4),
            "model": model_path.stem.removeprefix(f"{ticker}_"),
            "as_of": as_of,
        }


# ---------------------------------------------------------------- registry
class ServiceRegistry:
    """Owns the three lazy services and exposes health probing."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or repo_root()
        self._rag = RAGService(self._root)
        self._sms = SmsService(self._root)
        self._stock = StockService(self._root)
        self.started_at = time.monotonic()

    @property
    def rag(self) -> RAGService:
        return self._rag

    @property
    def sms(self) -> SmsService:
        return self._sms

    @property
    def stock(self) -> StockService:
        return self._stock

    def statuses(self) -> dict[str, str]:
        return {
            "rag": "available" if self._rag.available().available else "unavailable",
            "sms": "available" if self._sms.available().available else "unavailable",
            "stock": "available" if self._stock.available().available else "unavailable",
        }
