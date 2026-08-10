# 🎬 Multi-Domain Recommender System

![version](https://img.shields.io/badge/version-0.1.0-blue)
![python](https://img.shields.io/badge/python-3.9+-blue)
![status](https://img.shields.io/badge/status-active-brightgreen)

A collaborative-filtering recommender system built on a **domain-agnostic
architecture** — starting with movies (MovieLens 100K), with books and
music planned as drop-in extensions of the same `BaseRecommender` interface.

> This is Phase 1 (`v0.1.0`). See [Roadmap](#-roadmap) below for what's next.

## ✨ Features (Phase 1)

- **User-Based Collaborative Filtering** — recommends based on similar users' taste
- **Item-Based Collaborative Filtering** — recommends based on item-to-item similarity
- Sparse matrix implementation (`scipy.sparse`) — scales to large user/item counts
- Clean abstract interface (`BaseRecommender`) so future domains (books, music)
  only need to implement `fit` / `predict` / `recommend`
- Evaluation with RMSE, MAE, Precision@K, Recall@K

## 📁 Project Structure

```
recommender-system/
├── data/                   # datasets (gitignored, auto-downloaded)
├── src/
│   ├── data_loader.py      # downloads & loads MovieLens 100K
│   ├── base_recommender.py # abstract interface, shared eval logic
│   └── collaborative.py    # UserBasedCF, ItemBasedCF
├── tests/
│   └── test_collaborative.py
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/Abolfazlrwm/Machine-Learning.git
cd Machine-Learning/recommender-system
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Downloads MovieLens 100K automatically on first run
python -m src.data_loader
```

```python
from src.data_loader import load_ratings
from src.collaborative import ItemBasedCF

ratings = load_ratings()
model = ItemBasedCF(k_neighbors=30).fit(ratings)

print(model.recommend(user_id=1, n=10))
```

## 🧪 Running Tests & Evaluation

```bash
pytest tests/ -v -s
```

The evaluation test prints RMSE, MAE, Precision@10, and Recall@10 for
both algorithms on an 80/20 train/test split of MovieLens 100K.

## 📊 Results

| Model          | RMSE | MAE | Precision@10 | Recall@10 |
|----------------|------|-----|--------------|-----------|
| User-Based CF  | _run tests to fill in_ | | | |
| Item-Based CF  | _run tests to fill in_ | | | |

## 🗺 Roadmap

- [x] **v0.1.0** — MovieLens collaborative filtering (user-based + item-based)
- [ ] **v0.2.0** — Add books domain (Goodbooks-10k) via the same `BaseRecommender`
- [ ] **v0.2.0** — Add music domain (Last.fm)
- [ ] **v0.3.0** — Matrix Factorization (SVD) for better accuracy at scale
- [ ] **v0.3.0** — Interactive Streamlit demo (pick a domain, rate a few items, get recommendations)
- [ ] **v1.0.0** — Deployed live demo + polished docs

## 🛠 Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy (sparse matrices) · Streamlit (planned)
