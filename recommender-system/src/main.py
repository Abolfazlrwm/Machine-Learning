"""End-to-end demo: load MovieLens, train four from-scratch recommenders
(user-CF, item-CF, ALS, SVD), evaluate on a held-out split and print
sample recommendations for one user.

Run from the recommender-system/ directory (downloads the dataset on
first run):
    python -m src.main
"""
from __future__ import annotations

import time

import numpy as np

from src.collaborative import ItemBasedCF, UserBasedCF
from src.data_loader import load_movies, load_ratings, train_test_split_by_user
from src.evaluate import evaluate
from src.matrix_factorization import ALSRecommender, SVDRecommender

# Ranking metrics call recommend() once per user, and each CF call scans
# every candidate item — evaluating all 610 users would take hours. A
# fixed, seeded sample of users keeps the table honest and fast.
PR_USERS = 100


def titles_for(movies_df, item_ids):
    lookup = movies_df.set_index("item_id")["title"]
    return [lookup.get(m, f"(unknown movie {m})") for m in item_ids]


def main() -> None:
    print("Loading MovieLens ml-latest-small ...")
    ratings = load_ratings(download=True)
    movies = load_movies(download=True)
    print(f"  {len(ratings):,} ratings, {ratings.user_id.nunique():,} users, "
          f"{ratings.item_id.nunique():,} movies")

    train, test = train_test_split_by_user(ratings, test_size=0.2)
    print(f"  train: {len(train):,} ratings | test: {len(test):,} ratings\n")

    models = [
        ("User-Based CF", lambda: UserBasedCF(k_neighbors=30)),
        ("Item-Based CF", lambda: ItemBasedCF(k_neighbors=30)),
        ("ALS matrix factorization", lambda: ALSRecommender(n_factors=20, n_iterations=15, reg=0.05)),
        ("Truncated SVD", lambda: SVDRecommender(n_factors=15)),
    ]

    fitted = {}
    pr_sample = test[test.user_id.isin(
        np.random.default_rng(42).choice(test.user_id.unique(), PR_USERS, replace=False))]
    header = f"  {'model':26} {'RMSE':>6} {'MAE':>6} {'P@10':>6} {'R@10':>6} {'fit_s':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, factory in models:
        t0 = time.time()
        model = factory().fit(train)
        elapsed = time.time() - t0
        metrics = evaluate(model, test)
        pr = model.precision_recall_at_k(pr_sample, k=10)
        print(f"  {name:26} {metrics['rmse']:6.3f} {metrics['mae']:6.3f} "
              f"{pr['precision_at_k']:6.3f} {pr['recall_at_k']:6.3f} {elapsed:6.1f}",
              flush=True)
        fitted[name] = model

    demo_user = int(ratings.user_id.iloc[0])
    print(f"\nTop 5 recommendations for user {demo_user}:")
    for name in ("User-Based CF", "ALS matrix factorization"):
        print(f"  [{name}]")
        for item_id in fitted[name].recommend(demo_user, n=5):
            print(f"    {titles_for(movies, [item_id])[0]}")


if __name__ == "__main__":
    main()
