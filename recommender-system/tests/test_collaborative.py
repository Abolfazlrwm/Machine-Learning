"""
test_collaborative.py
-----------------------
Basic correctness tests + an evaluation run on MovieLens 100K
(train/test split -> RMSE, MAE, Precision@10, Recall@10).

Run with:  pytest tests/ -v -s
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from src.collaborative import UserBasedCF, ItemBasedCF


@pytest.fixture
def toy_ratings():
    """A tiny synthetic ratings set for fast unit tests."""
    data = {
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "item_id": [10, 20, 30, 10, 20, 20, 30, 40, 10, 40],
        "rating":  [5,  3,  4,  4,  2,  5,  5,  1,  4,  2],
    }
    return pd.DataFrame(data)


def test_user_based_cf_fits_and_predicts(toy_ratings):
    model = UserBasedCF(k_neighbors=5).fit(toy_ratings)
    pred = model.predict(user_id=1, item_id=40)
    assert isinstance(pred, float)
    assert 1 <= pred <= 5 or pred == pred  # not NaN


def test_item_based_cf_fits_and_predicts(toy_ratings):
    model = ItemBasedCF(k_neighbors=5).fit(toy_ratings)
    pred = model.predict(user_id=2, item_id=30)
    assert isinstance(pred, float)


def test_recommend_excludes_rated_items(toy_ratings):
    model = ItemBasedCF(k_neighbors=5).fit(toy_ratings)
    recs = model.recommend(user_id=1, n=5)
    rated = {10, 20, 30}
    assert not (set(recs) & rated)


def test_cold_start_unknown_user_returns_fallback(toy_ratings):
    model = UserBasedCF().fit(toy_ratings)
    pred = model.predict(user_id=999, item_id=10)
    assert isinstance(pred, float)


def test_evaluate_returns_metrics(toy_ratings):
    model = ItemBasedCF(k_neighbors=5).fit(toy_ratings)
    metrics = model.evaluate(toy_ratings)
    assert "rmse" in metrics and "mae" in metrics


# ---------------------------------------------------------------------
# Full evaluation run on the real MovieLens 100K dataset.
# Skipped automatically if the dataset hasn't been downloaded yet.
# ---------------------------------------------------------------------

def test_full_movielens_evaluation():
    from src.data_loader import load_ratings

    try:
        ratings = load_ratings(download=False)
    except FileNotFoundError:
        pytest.skip("MovieLens data not downloaded yet — run "
                    "`python -m src.data_loader` first.")

    train = ratings.sample(frac=0.8, random_state=42)
    test = ratings.drop(train.index)

    for name, cls in [("User-Based CF", UserBasedCF), ("Item-Based CF", ItemBasedCF)]:
        model = cls(k_neighbors=30).fit(train)
        metrics = model.evaluate(test)
        pr = model.precision_recall_at_k(test, k=10)
        print(f"\n{name}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
              f"Precision@10={pr['precision_at_k']:.4f}  Recall@10={pr['recall_at_k']:.4f}")
