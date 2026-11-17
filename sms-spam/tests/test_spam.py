"""Unit tests for the sms-spam package — synthetic, deterministic.

`clean_text` needs the two small NLTK datasets (stopwords, wordnet);
the CI job downloads them in a dedicated step before running pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import predict_message  # noqa: E402
from preprocess import clean_text, ensure_nltk  # noqa: E402
from train import PARAM_GRID, build_pipeline, load_dataset, train_model  # noqa: E402

ensure_nltk()


def test_clean_text_strips_noise() -> None:
    cleaned = clean_text("WIN a FREE prize! Call http://spam.xyz 0780... now")
    assert "http" not in cleaned
    assert "0780" not in cleaned
    assert "!" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_stopwords_and_lemmatization() -> None:
    assert clean_text("the dogs were running") == "dog running"
    assert clean_text("Hello, World!") == "hello world"


def test_clean_text_is_deterministic() -> None:
    text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May"
    assert clean_text(text) == clean_text(text)


def test_load_dataset_maps_labels(tmp_path: Path) -> None:
    csv = tmp_path / "tiny.csv"
    frame = pd.DataFrame({"v1": ["ham", "spam", "ham"],
                          "v2": ["hi there", "win now", "ok later"],
                          "unnamed": ["x", "y", "z"]})
    frame.to_csv(csv, index=False, encoding="latin-1")
    df = load_dataset(csv)
    assert list(df["label"]) == [0, 1, 0]
    assert "cleaned" in df.columns


def test_build_pipeline_and_param_grid() -> None:
    pipe = build_pipeline()
    assert [name for name, _ in pipe.steps] == ["tfidf", "clf"]
    tfidf = pipe.named_steps["tfidf"]
    assert tfidf.max_features == 5000
    assert tfidf.ngram_range == (1, 2)
    assert set(PARAM_GRID) == {"clf__C", "clf__solver"}


def _synthetic_csv(tmp_path: Path, n_repeats: int = 20) -> Path:
    """Two obviously separable message families — enough for the pipeline
    to learn, small enough to train in well under a second."""
    spam = "urgent win free prize claim now txt cash bonus call"
    ham = "hey how are you lunch tomorrow meeting ok later love"
    rows = [{"v1": "spam", "v2": spam} for _ in range(n_repeats)]
    rows += [{"v1": "ham", "v2": ham} for _ in range(n_repeats)]
    csv = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(csv, index=False, encoding="latin-1")
    return csv


def test_train_predict_roundtrip(tmp_path: Path) -> None:
    csv = _synthetic_csv(tmp_path)
    model_out = tmp_path / "artifacts" / "model.pkl"
    metrics = train_model(csv, model_out=model_out)
    assert model_out.exists()
    assert metrics["accuracy"] >= 0.9
    assert set(metrics) == {"best_params", "accuracy", "roc_auc", "cv_accuracy",
                            "spam_precision", "spam_recall", "spam_f1"}
    assert predict_message("win free prize claim now", model_out) == "SPAM"
    assert predict_message("hey how are you", model_out) == "HAM (Not Spam)"


def test_predict_without_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train first"):
        predict_message("hello", tmp_path / "missing.pkl")
