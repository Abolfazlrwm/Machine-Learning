# Recommender system

Four recommenders built **from first principles** on MovieLens
`ml-latest-small` (100,836 ratings, 610 users, 9,724 movies) — no
recommendation library (Surprise, LightFM, implicit): similarities,
predictions, matrix factorization and evaluation are all implemented
directly with pandas/NumPy.

## Models

**User-Based CF** — predict user U's rating for item M from the k most
similar users (cosine similarity on *mean-centered* rating vectors,
k=30). Centering matters: comparing raw vectors treats "not rated" as
"disliked" and punishes generous raters; the implementation centres per
user and keeps unrated cells at zero.

**Item-Based CF** — similarity between items over the whole userbase
(k=30); a user's own past ratings for similar items predict the new one.
The same principle behind "customers who bought this also bought."

**ALS matrix factorization** — `R ≈ PᵀQ` with alternating least squares:
each half-step is an exact ridge solve per user / per item (20 latent
factors, 15 iterations, L2 0.05), with a bias column of ones folded
into the factors so intercepts come out of the closed form. Cold-start
users/items fall back to a user-mean → item-mean → global-mean ladder.

**Truncated SVD** — the classic baseline: fill the sparse matrix with
the global mean, keep rank 15 of the SVD, reconstruct. Deterministic,
no initialisation randomness — a useful quality reference for ALS.

## Results (80/20 per-user held-out split)

| Model | RMSE | MAE | P@10 | R@10 | fit time |
|---|---|---|---|---|---|
| User-Based CF (centered, k=30) | 0.885 | 0.674 | 0.002 | 0.001 | 0.5 s |
| Item-Based CF (k=30) | 0.875 | 0.667 | 0.011 | 0.001 | 1.0 s |
| **ALS (20 factors)** | **0.914** | **0.701** | 0.026 | 0.011 | 6.7 s |
| Truncated SVD (rank 15) | 0.992 | 0.782 | 0.157 | 0.132 | 1.8 s |

Honest reading:

* **Item-based CF wins on rating accuracy** (0.875) — the matrix is dense
  enough (~169 ratings/user) that neighbourhood methods beat an un-tuned
  factorization; ALS still lands inside the published 0.87–0.95 baseline
  band for MovieLens 100k without any hyperparameter search.
* **SVD's high P@10 (0.157) is a popularity artifact, not a win**: its
  reconstruction of the global-mean-filled matrix ranks mainstream items
  on top — a textbook reminder that ranking metrics and rating metrics
  rank models *differently*.
* Ranking metrics are averaged over a fixed 100-user sample (calling
  `recommend()` per user re-scores every candidate; all 610 users would
  take hours). Rating metrics cover the full 20,164-row held-out set.

Published baselines for classic CF on MovieLens 100k sit around
RMSE 0.87–0.95, so these un-tuned from-scratch numbers are in range.

## Reproduce

```bash
cd recommender-system
pip install -r requirements.txt
python -m src.main          # downloads the dataset on first run (~1 MB)
python -m pytest tests/ -q  # 25 tests — synthetic unit tests + a real-data test that skips without the CSVs
```
