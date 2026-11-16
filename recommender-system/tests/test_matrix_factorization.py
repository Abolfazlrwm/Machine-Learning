"""Matrix factorization tests — synthetic, deterministic, CI-safe.

ALS is checked against ground truth twice: *exact* recovery of a
noise-free low-rank matrix, and a no-noise-information win over the
mean baseline on a noisy one. SVD is checked on rank-1 exactness.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.base_recommender import BaseRecommender
from src.matrix_factorization import ALSRecommender, SVDRecommender


def make_low_rank(n_users=12, n_items=15, rank=3, noise=0.0, seed=7):
    """Low-rank + constant ratings whose values stay inside [0.5, 5.0]
    *without* clipping — clipping would destroy the exact low-rank
    structure the recovery tests need."""
    rng = np.random.default_rng(seed)
    p = rng.normal(scale=0.35, size=(rank, n_users))
    q = rng.normal(scale=0.35, size=(rank, n_items))
    true = 3.5 + p.T @ q
    if noise:
        true = true + rng.normal(scale=noise, size=true.shape)
    return true


def to_dataframe(matrix: np.ndarray, frac: float = 1.0, seed: int = 3) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for u in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            if rng.random() <= frac:
                rows.append({"user_id": u, "item_id": i, "rating": matrix[u, i]})
    return pd.DataFrame(rows)


@pytest.fixture
def dense_low_rank():
    return make_low_rank(noise=0.0)


@pytest.fixture
def sparse_noisy():
    m = make_low_rank(n_users=20, n_items=25, rank=3, noise=0.4, seed=11)
    return to_dataframe(m, frac=0.3, seed=12)


class TestALS:
    def test_is_a_base_recommender(self):
        assert isinstance(ALSRecommender(), BaseRecommender)

    def test_exact_recovery_of_noise_free_low_rank(self, dense_low_rank):
        df = to_dataframe(dense_low_rank)
        model = ALSRecommender(n_factors=3, n_iterations=30, reg=1e-6, seed=1).fit(df)
        preds = np.array([
            model.predict(int(r.user_id), int(r.item_id))
            for r in df.itertuples(index=False)
        ])
        rmse = float(np.sqrt(np.mean((preds - df.rating.values) ** 2)))
        assert rmse < 0.01, f"ALS failed to recover a rank-3 matrix: rmse={rmse:.4f}"

    def test_beats_mean_baseline_on_noisy_data(self, sparse_noisy):
        model = ALSRecommender(n_factors=5, n_iterations=20, reg=0.1, seed=2).fit(sparse_noisy)
        als_rmse = model.evaluate(sparse_noisy)["rmse"]
        baseline = float(np.sqrt(np.mean((sparse_noisy.rating - sparse_noisy.rating.mean()) ** 2)))
        assert als_rmse < baseline, f"ALS {als_rmse:.3f} vs mean baseline {baseline:.3f}"

    def test_train_rmse_decreases(self, sparse_noisy):
        model = ALSRecommender(n_factors=5, n_iterations=10, reg=0.1, seed=3).fit(sparse_noisy)
        assert model.train_rmse_history[-1] < model.train_rmse_history[0]

    def test_deterministic_for_fixed_seed(self, sparse_noisy):
        a = ALSRecommender(n_factors=5, n_iterations=5, seed=4).fit(sparse_noisy)
        b = ALSRecommender(n_factors=5, n_iterations=5, seed=4).fit(sparse_noisy)
        assert np.allclose(a.user_factors, b.user_factors)
        assert a.predict(1, 1) == b.predict(1, 1)

    def test_predictions_stay_on_rating_scale(self, sparse_noisy):
        model = ALSRecommender(n_factors=5, n_iterations=10, seed=5).fit(sparse_noisy)
        for r in sparse_noisy.itertuples(index=False):
            assert 0.5 <= model.predict(int(r.user_id), int(r.item_id)) <= 5.0

    def test_cold_start_fallbacks(self, sparse_noisy):
        model = ALSRecommender(n_factors=5, seed=6).fit(sparse_noisy)
        known_user = int(sparse_noisy.user_id.iloc[0])
        assert model.predict(known_user, 10_000) == pytest.approx(
            float(model.user_means[model.user_index[known_user]]))
        assert model.predict(99_999, 10_000) == pytest.approx(model.global_mean)

    def test_recommend_excludes_rated_and_returns_n(self, sparse_noisy):
        model = ALSRecommender(n_factors=5, seed=7).fit(sparse_noisy)
        user = int(sparse_noisy.user_id.iloc[0])
        rated = set(sparse_noisy[sparse_noisy.user_id == user].item_id)
        recs = model.recommend(user, n=5)
        assert len(recs) == 5
        assert all(isinstance(item, int) for item in recs)
        assert all(item not in rated for item in recs)
        assert all(0.5 <= model.predict(user, item) <= 5.0 for item in recs)

    def test_unfitted_model_raises(self):
        with pytest.raises(RuntimeError):
            ALSRecommender().recommend(1)


class TestSVD:
    def test_rank1_exact_recovery(self):
        rng = np.random.default_rng(8)
        u = rng.normal(scale=0.3, size=(10, 1))
        v = rng.normal(scale=0.3, size=(1, 12))
        matrix = 3.5 + u @ v  # stays inside the rating scale without clipping
        df = to_dataframe(matrix)
        # 3.5*ones is itself rank-1, so the matrix is rank 2 in total
        model = SVDRecommender(n_factors=2).fit(df)
        preds = np.array([
            model.predict(int(r.user_id), int(r.item_id))
            for r in df.itertuples(index=False)
        ])
        rmse = float(np.sqrt(np.mean((preds - df.rating.values) ** 2)))
        assert rmse < 0.01

    def test_deterministic(self, sparse_noisy):
        a = SVDRecommender(n_factors=5).fit(sparse_noisy)
        b = SVDRecommender(n_factors=5).fit(sparse_noisy)
        assert np.allclose(a.reconstruction, b.reconstruction)

    def test_beats_mean_baseline(self, sparse_noisy):
        model = SVDRecommender(n_factors=8).fit(sparse_noisy)
        svd_rmse = model.evaluate(sparse_noisy)["rmse"]
        baseline = float(np.sqrt(np.mean((sparse_noisy.rating - sparse_noisy.rating.mean()) ** 2)))
        assert svd_rmse < baseline
