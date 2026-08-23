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

## Results

Evaluated with an 80/20 per-user train/test split (so every user is
represented in training — a plain random split can accidentally drop
a user entirely and break CF for them):

| Model         | RMSE  | MAE   |
|---------------|-------|-------|
| User-Based CF | 0.877 | 0.668 |
| Item-Based CF | 0.925 | 0.709 |

For reference, these are in the same range as published baselines for
classic CF on MovieLens 100k (typically RMSE ≈ 0.87–0.95), which is
what you'd expect from a correct, un-tuned implementation.

## Project structure

```
recommender-system/
├── data/              MovieLens ml-latest-small (ratings, movies, links, tags)
├── src/
│   ├── data_loader.py     CSV loading + user-item matrix + train/test split
│   ├── user_based_cf.py   User-Based CF
│   ├── item_based_cf.py   Item-Based CF
│   ├── evaluate.py        RMSE / MAE
│   └── main.py            End-to-end demo
└── tests/
    └── test_recommenders.py   7 unit tests on a hand-built toy matrix
```

## Run it

```bash
pip install -r requirements.txt
cd src && python main.py
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
pytest tests/ -v
```

## Possible next steps

- Matrix Factorization (SVD/ALS) baseline for comparison
- Hybrid: blend user-based + item-based predictions
- Extend beyond movies to books/music (multi-domain, as originally scoped)
- Cold-start handling via genre-based content similarity
