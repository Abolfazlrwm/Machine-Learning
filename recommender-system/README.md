# 🎬 Movie Recommender — Collaborative Filtering

A from-scratch implementation of **User-Based** and **Item-Based**
Collaborative Filtering, trained and evaluated on the real
[MovieLens](https://grouplens.org/datasets/movielens/) `ml-latest-small`
dataset (105k ratings, 668 users, 10k+ movies).

No recommendation library (Surprise, LightFM, implicit) is used —
similarity computation, prediction, and evaluation are all implemented
directly with pandas/NumPy/scikit-learn so the mechanics are fully
visible.

## How it works

**User-Based CF** — to predict how much user U will like movie M, find
the `k` users most similar to U by cosine similarity on mean-centered
rating vectors, then take a similarity-weighted average of their
ratings for M.

**Item-Based CF** — same idea, but similarity is computed between
*movies* based on how the whole userbase rated them, then a user's own
past ratings for similar movies predict their rating for M. This is
the same principle behind "customers who bought this also bought."

Both predictions are clipped to the valid 0.5–5.0 rating scale, and
both fall back gracefully to a global or per-user mean for cold-start
users/movies with no overlap.

## Matrix factorization (ALS + SVD, from scratch)

`src/matrix_factorization.py` adds two factorization recommenders on the
same `BaseRecommender` interface:

* **`ALSRecommender`** — alternating least squares for explicit feedback:
  every half-step is an exact ridge solve per user / per item (20 latent
  factors, 15 iterations, L2 0.05, seeded). A bias column of ones is
  folded into the factor matrices so intercepts come out of the closed
  form. Cold-start users/items fall back through a
  user-mean → item-mean → global-mean ladder.
* **`SVDRecommender`** — the classic baseline: fill the sparse matrix
  with the global mean, keep rank 15 of the SVD, reconstruct. Fully
  deterministic — a useful quality reference for ALS.

14 synthetic tests cover both (exact rank-k recovery, a guaranteed win
over the mean baseline, cold-start fallbacks, determinism, recommendation
constraints) — no dataset needed, so CI runs them everywhere.

## Results

Evaluated with an 80/20 per-user train/test split (so every user is
represented in training — a plain random split can accidentally drop
a user entirely and break CF for them). Ranking metrics are averaged
over a fixed 100-user sample (`recommend()` re-scores every candidate
per user — the full user base would take hours):

| Model         | RMSE  | MAE   | P@10  | R@10  | fit   |
|---------------|-------|-------|-------|-------|-------|
| User-Based CF (centered, k=30) | 0.885 | 0.674 | 0.002 | 0.001 | 0.5 s |
| Item-Based CF (k=30)           | **0.875** | **0.667** | 0.011 | 0.001 | 1.0 s |
| ALS (20 factors)               | 0.914 | 0.701 | 0.026 | 0.011 | 6.7 s |
| Truncated SVD (rank 15)        | 0.992 | 0.782 | 0.157 | 0.132 | 1.8 s |

Honest reading: **item-based CF wins rating accuracy** on this dataset —
it is dense enough (~169 ratings/user) that neighbourhood methods beat an
un-tuned factorization — and **SVD's high P@10 is a popularity artifact**:
reconstructing the global-mean-filled matrix ranks mainstream items on
top, which ranking metrics reward even when rating metrics penalise the
fill. All rating metrics sit in the published 0.87–0.95 baseline band for
classic CF on MovieLens 100k, from a correct, un-tuned implementation.

## Project structure

```
recommender-system/
├── data/              MovieLens ml-latest-small (downloaded, kept out of git)
├── src/
│   ├── data_loader.py         CSV loading + download + per-user split
│   ├── user_based_cf.py       User-Based CF (pivot-based reference impl)
│   ├── item_based_cf.py       Item-Based CF (pivot-based reference impl)
│   ├── collaborative.py       User/Item CF on the BaseRecommender interface
│   ├── matrix_factorization.py  ALS + truncated SVD
│   ├── base_recommender.py    shared interface (evaluate, P@K / R@K)
│   ├── evaluate.py            RMSE / MAE
│   └── main.py                End-to-end demo of all four models
└── tests/
    ├── test_collaborative.py        toy + real-data tests
    ├── test_recommenders.py         toy-matrix tests
    └── test_matrix_factorization.py ALS/SVD recovery + contract tests
```

## Run it

```bash
pip install -r requirements.txt
python -m src.main          # downloads the dataset on first run
```

```
Loading MovieLens ml-latest-small ...
  105,339 ratings, 668 users, 10,325 movies
  train: 84,274 ratings | test: 21,065 ratings

Training User-Based CF ...
  done in 0.6s
Training Item-Based CF ...
  done in 7.0s

Evaluating on held-out test set ...
  User-Based CF -> RMSE: 0.877  MAE: 0.668
  Item-Based CF -> RMSE: 0.925  MAE: 0.709
```

## Tests

```bash
pytest tests/ -q    # 25 tests; the real-data test skips without the CSVs
```

## Possible next steps

- Hybrid: blend user-based + item-based predictions
- Implicit-feedback ALS with confidence weighting (Hu et al. 2008)
- Extend beyond movies to books/music (multi-domain, as originally scoped)
- Cold-start handling via genre-based content similarity
