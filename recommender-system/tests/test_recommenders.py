import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from user_based_cf import UserBasedCF
from item_based_cf import ItemBasedCF


@pytest.fixture
def toy_matrix():
    """A small, hand-built user x movie matrix with an obvious pattern:
    users 1 and 2 have near-identical taste, user 3 is the odd one out.
    """
    data = {
        101: [5.0, 4.5, 1.0],
        102: [4.0, 4.0, 1.5],
        103: [1.0, 1.5, 5.0],
        104: [np.nan, 5.0, np.nan],  # only user 2 has rated this one
    }
    return pd.DataFrame(data, index=[1, 2, 3])


def test_user_based_predicts_within_rating_scale(toy_matrix):
    model = UserBasedCF(k_neighbors=2).fit(toy_matrix)
    pred = model.predict(1, 104)
    assert 0.5 <= pred <= 5.0


def test_user_based_finds_similar_taste(toy_matrix):
    # Users 1 and 2 both love 101/102 and dislike 103, so user 1's
    # unknown rating for 104 should be pulled toward user 2's (5.0),
    # not toward user 3's taste.
    model = UserBasedCF(k_neighbors=2).fit(toy_matrix)
    pred = model.predict(1, 104)
    assert pred > 3.0


def test_item_based_predicts_within_rating_scale(toy_matrix):
    model = ItemBasedCF(k_neighbors=2).fit(toy_matrix)
    pred = model.predict(3, 104)
    assert 0.5 <= pred <= 5.0


def test_unknown_user_falls_back_to_global_mean(toy_matrix):
    model = UserBasedCF().fit(toy_matrix)
    pred = model.predict(999, 101)
    assert pred == pytest.approx(model.user_means.mean())


def test_unknown_movie_falls_back_to_user_mean(toy_matrix):
    model = UserBasedCF().fit(toy_matrix)
    pred = model.predict(1, 999)
    assert pred == pytest.approx(model.user_means.loc[1])


def test_recommend_excludes_already_rated_movies(toy_matrix):
    model = UserBasedCF(k_neighbors=2).fit(toy_matrix)
    recs = model.recommend(1, n=10)
    recommended_ids = [m for m, _ in recs]
    assert 101 not in recommended_ids  # user 1 already rated this
    assert 104 in recommended_ids      # user 1 hasn't rated this


def test_recommend_returns_empty_for_unknown_user(toy_matrix):
    model = UserBasedCF().fit(toy_matrix)
    assert model.recommend(999) == []
