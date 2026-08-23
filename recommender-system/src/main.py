"""End-to-end demo: load MovieLens, train both CF models, evaluate on
a held-out split, and print sample recommendations for one user.

Run from the recommender-system/ directory:
    python src/main.py
"""
from __future__ import annotations

import time

from data_loader import load_ratings, load_movies, build_user_item_matrix, train_test_split_by_user
from user_based_cf import UserBasedCF
from item_based_cf import ItemBasedCF
from evaluate import evaluate


def titles_for(movies_df, movie_ids):
    lookup = movies_df.set_index("movieId")["title"]
    return [lookup.get(m, f"(unknown movie {m})") for m in movie_ids]


def main() -> None:
    print("Loading MovieLens ml-latest-small ...")
    ratings = load_ratings()
    movies = load_movies()
    print(f"  {len(ratings):,} ratings, {ratings.userId.nunique():,} users, "
          f"{ratings.movieId.nunique():,} movies")

    train, test = train_test_split_by_user(ratings, test_size=0.2)
    print(f"  train: {len(train):,} ratings | test: {len(test):,} ratings\n")

    user_item = build_user_item_matrix(train)

    print("Training User-Based CF ...")
    t0 = time.time()
    user_cf = UserBasedCF(k_neighbors=30).fit(user_item)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Training Item-Based CF ...")
    t0 = time.time()
    item_cf = ItemBasedCF(k_neighbors=30).fit(user_item)
    print(f"  done in {time.time() - t0:.1f}s\n")

    print("Evaluating on held-out test set ...")
    user_metrics = evaluate(user_cf, test)
    print(f"  User-Based CF -> RMSE: {user_metrics['rmse']:.3f}  MAE: {user_metrics['mae']:.3f}")

    item_metrics = evaluate(item_cf, test)
    print(f"  Item-Based CF -> RMSE: {item_metrics['rmse']:.3f}  MAE: {item_metrics['mae']:.3f}\n")

    demo_user = int(user_item.index[0])
    print(f"Top 5 recommendations for user {demo_user} (User-Based CF):")
    for movie_id, score in user_cf.recommend(demo_user, n=5):
        title = titles_for(movies, [movie_id])[0]
        print(f"  {score:.2f}  {title}")

    print(f"\nTop 5 recommendations for user {demo_user} (Item-Based CF):")
    for movie_id, score in item_cf.recommend(demo_user, n=5):
        title = titles_for(movies, [movie_id])[0]
        print(f"  {score:.2f}  {title}")


if __name__ == "__main__":
    main()
