# Machine Learning

[![Tests](https://github.com/Abolfazlrwm/Machine-Learning/actions/workflows/tests.yml/badge.svg)](https://github.com/Abolfazlrwm/Machine-Learning/actions/workflows/tests.yml)

A collection of end-to-end machine learning projects — from classical
scikit-learn pipelines to LLM-powered retrieval. Every project ships with a
structured codebase (`src/` + `tests/`), real evaluation numbers and a
documented methodology.

## 🗂 Projects

| Project | Domain | Key techniques | Headline result |
|---|---|---|---|
| [`rag-farsi-assistant`](rag-farsi-assistant/) | LLM / Retrieval (RAG) | Persian RAG: E5 embeddings + FAISS **fused with a from-scratch BM25 via RRF**, switchable dense/sparse/hybrid modes, optional cross-encoder re-ranking, RAGAS-style eval suite, Ollama/OpenAI providers, Streamlit chat | 30-pair golden set: hybrid Hit@1 = 0.967, MRR = 0.983 (E5+BM25 via RRF) · context recall 0.98 · 72 tests |
| [`ai-text-detector`](ai-text-detector/) | NLP classification | TF-IDF + stylometric features (sentence length variation, punctuation habits), Linear SVM vs NB vs LR | Accuracy 0.897 · F1 0.899 · ROC-AUC 0.964 |
| [`recommender-system`](recommender-system/) | RecSys | User-based & item-based collaborative filtering **from scratch** (pandas/NumPy) on MovieLens 105k | User-CF RMSE 0.877 · Item-CF RMSE 0.925 |
| [`stock-movement-ml`](stock-movement-ml/) | Finance / time series | 20 technical indicators, walk-forward validation, leakage-free backtest with costs, Plotly dashboard | OOS ROC-AUC 0.53 (Random Forest), +25.45% on 696 out-of-sample days |
| [`sms-spam`](sms-spam/) | NLP baseline | TF-IDF (1–2 grams) + Logistic Regression, 5-fold GridSearchCV | Accuracy 0.975 · ROC-AUC 0.987 |

## 🧭 What each project demonstrates

- **rag-farsi-assistant** — the full modern RAG loop: chunking with overlap,
  multilingual embeddings (E5 with automatic query/passage prefixes), FAISS
  vector search, **hybrid retrieval that fuses dense + BM25 rankings with
  Reciprocal Rank Fusion (BM25 implemented from scratch, Persian-aware
  tokenisation)**, optional cross-encoder re-ranking, strict Persian prompts
  that force citation and "I don't know" behaviour, and a provider layer that
  swaps **Ollama (local, free) ⇄ any OpenAI-compatible API** with one line.
  Evaluated like production: a 30-pair golden set, dense-vs-sparse-vs-hybrid
  comparison and a deterministic RAGAS-style generation suite — hybrid
  retrieval wins with Hit@1 0.967 / MRR 0.983. Ships a
  Streamlit chat UI, CLI, and a dependency-free "fake" embedding backend for CI.
- **ai-text-detector** — classic feature engineering: does *style* reveal
  machine text better than *vocabulary*? Hybrid TF-IDF + stylometric features
  beat plain TF-IDF across all three classifiers.
- **recommender-system** — collaborative filtering implemented from first
  principles (similarity matrices, weighted predictions, RMSE/MAE evaluation),
  no recommendation libraries.
- **stock-movement-ml** — financial ML done honestly: walk-forward
  validation instead of a single split, signals aligned to next-day returns,
  transaction costs, and results reported *including* the uncomfortable ones
  (daily direction ≈ coin flip).

## 🚀 Quick start

Each project is self-contained with its own `requirements.txt` and README:

```bash
# RAG assistant (needs an Ollama model or an OpenAI-compatible key)
cd rag-farsi-assistant
pip install -r requirements.txt
python main.py build && python main.py ask "بیش‌برازش چیست؟"
streamlit run app.py

# AI text detector
cd ai-text-detector && pip install -r requirements.txt

# Recommender system
cd recommender-system && pip install -r requirements.txt

# Stock movement (walk-forward pipeline + dashboard)
cd stock-movement-ml && pip install -r requirements.txt
python main.py --ticker AAPL --start 2018-01-01 --end 2025-01-01

# SMS spam baseline
cd sms-spam && pip install pandas scikit-learn nltk joblib
python spam_classifier.py --train spam.csv
```

## 📦 Datasets

The datasets are not stored in git (they would bloat the repository) — fetch
them once from the repo root:

```bash
python download_data.py
```

| Project | Dataset | Source |
|---|---|---|
| `recommender-system` | MovieLens ml-latest-small (`ratings.csv`, `movies.csv`) | [GroupLens](https://grouplens.org/datasets/movielens/) |
| `ai-text-detector` | Human vs AI Generated Text (`Data`, `Labels`) | Kaggle — mirrored byte-for-byte as a [release asset](https://github.com/Abolfazlrwm/Machine-Learning/releases/tag/datasets-v1) |
| `sms-spam` | SMS Spam Collection (5,574 messages) | bundled at `sms-spam/spam.csv` |

Trained model artifacts (`*.joblib`, `*.pkl`) and generated reports
(`stock-movement-ml/reports/`) are git-ignored as well — recreate them by
running each project's own training pipeline.

## 🧪 Testing

Projects include pytest suites (unit-level, network-free):

```bash
cd rag-farsi-assistant && python -m pytest tests/ -q   # 72 tests
cd stock-movement-ml  && python -m pytest tests/ -q    # 27 tests
```

## 🛣 Roadmap

- [x] Hybrid retrieval (BM25 + RRF) and a RAGAS-style evaluation suite for the RAG assistant — v2.1.0
- [ ] Matrix factorization (SVD/ALS) for the recommender system
- [ ] PDF ingestion for the RAG assistant
- [ ] Fine-tuning a Persian BERT for sentiment analysis
- [ ] Hyperparameter tuning with Optuna for the stock pipeline

## 🏷 Releases & changelog

Releases are tagged per milestone — see [CHANGELOG.md](CHANGELOG.md) and the
[releases page](https://github.com/Abolfazlrwm/Machine-Learning/releases):

| Tag | Scope |
|---|---|
| `v2.1.0` · `rag-v1.1.0` | Hybrid search (BM25 + RRF), RAGAS-style eval, 30-pair golden set |
| `v2.0.0` | Portfolio-wide CI, README overhaul, project restructure |
| `datasets-v1` | Datasets mirrored as release assets (kept out of git) |
