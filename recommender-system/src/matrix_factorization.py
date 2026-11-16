"""Matrix factorization recommenders — ALS and truncated SVD, from scratch.

Both factor the user-item rating matrix ``R ≈ PᵀQ`` and complete it;
neither uses a recommendation library.

``ALSRecommender`` — alternating least squares for explicit feedback.
A bias column of ones is folded into the factor matrices, so the ridge
closed form learns user/item intercepts without a separate bias step.
Each half-step solves one independent ridge regression per user (or
item) — embarrassingly parallel in spirit, exact in arithmetic.

``SVDRecommender`` — the classic baseline: fill the sparse matrix with
the global mean, truncate the SVD to rank k, reconstruct. Deterministic
(no initialisation randomness) and a useful quality reference for ALS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.base_recommender import BaseRecommender

MIN_RATING, MAX_RATING = 0.5, 5.0


class _MFBase(BaseRecommender):
    """Shared machinery: id lookups, dense rating matrix, mean fallbacks."""

    def __init__(self):
        self.user_ids = None
        self.item_ids = None
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.matrix: np.ndarray | None = None       # (n_users, n_items), NaN = unrated
        self.global_mean = 0.0
        self.user_means = None
        self.item_means = None

    def _setup(self, ratings_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.user_ids = np.sort(ratings_df.user_id.unique())
        self.item_ids = np.sort(ratings_df.item_id.unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}

        matrix = np.full((len(self.user_ids), len(self.item_ids)), np.nan)
        for row in ratings_df.itertuples(index=False):
            matrix[self.user_index[row.user_id], self.item_index[row.item_id]] = row.rating

        self.global_mean = float(np.nanmean(matrix))
        self.user_means = np.nanmean(matrix, axis=1)
        self.item_means = np.nanmean(matrix, axis=0)
        self.user_means = np.where(np.isnan(self.user_means), self.global_mean, self.user_means)
        self.item_means = np.where(np.isnan(self.item_means), self.global_mean, self.item_means)
        self.matrix = matrix
        return matrix

    def _known(self, user_id: int, item_id: int) -> tuple[int | None, int | None]:
        u = self.user_index.get(user_id)
        i = self.item_index.get(item_id)
        return u, i

    def _fallback(self, user_id: int, item_id: int,
                  u: int | None, i: int | None) -> float:
        """Cold-start ladder: user mean > item mean > global mean."""
        if u is not None and i is not None:
            return self.global_mean
        if u is not None:
            return float(self.user_means[u])
        if i is not None:
            return float(self.item_means[i])
        return self.global_mean

    def _clip(self, value: float) -> float:
        return float(min(max(value, MIN_RATING), MAX_RATING))

    def recommend(self, user_id: int, n: int = 10) -> list:
        """Top-N item ids for a user (the BaseRecommender contract)."""
        if self.matrix is None:
            raise RuntimeError("model is not fitted")
        u, _ = self._known(user_id, -1)
        if u is None:
            # fully cold-started user: the most-rated items stand in for taste
            counts = np.sum(~np.isnan(self.matrix), axis=0)
            order = np.argsort(-counts)
            return [int(self.item_ids[i]) for i in order[:n]]
        rated = np.isnan(self.matrix[u])
        scores = np.array([self.predict(user_id, int(item)) for item in self.item_ids])
        scores[~rated] = -np.inf
        order = np.argsort(-scores)[:n]
        return [int(self.item_ids[i]) for i in order]


class ALSRecommender(_MFBase):
    """Alternating least squares with folded-in bias, ridge-regularised."""

    def __init__(self, n_factors: int = 20, n_iterations: int = 15,
                 reg: float = 0.05, seed: int = 42):
        super().__init__()
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.seed = seed
        self.user_factors = None   # (f+1, n_users) — last row is the bias
        self.item_factors = None   # (f+1, n_items)
        self.train_rmse_history: list[float] = []

    def fit(self, ratings_df: pd.DataFrame) -> "ALSRecommender":
        matrix = self._setup(ratings_df)
        observed = ~np.isnan(matrix)
        target = np.where(observed, matrix, 0.0)

        rng = np.random.default_rng(self.seed)
        f = self.n_factors + 1                    # + bias column
        item_factors = rng.normal(scale=0.1, size=(f, matrix.shape[1]))
        item_factors[-1, :] = self.item_means - self.global_mean
        eye = np.eye(f)

        for _ in range(self.n_iterations):
            # user half-step: one ridge solve per user
            user_factors = np.zeros((f, matrix.shape[0]))
            for u in range(matrix.shape[0]):
                idx = observed[u]
                if not idx.any():
                    user_factors[:, u] = 0.0
                    user_factors[-1, u] = self.user_means[u] - self.global_mean
                    continue
                m = item_factors[:, idx]                      # (f, n_items_u)
                r = target[u, idx] - self.global_mean         # centred ratings
                a = m @ m.T + self.reg * idx.sum() * eye
                user_factors[:, u] = np.linalg.solve(a, m @ r)
            # item half-step: symmetric
            new_items = np.zeros((f, matrix.shape[1]))
            for i in range(matrix.shape[1]):
                idx = observed[:, i]
                if not idx.any():
                    new_items[:, i] = item_factors[:, i]
                    continue
                p = user_factors[:, idx]
                r = target[idx, i] - self.global_mean
                a = p @ p.T + self.reg * idx.sum() * eye
                new_items[:, i] = np.linalg.solve(a, p @ r)
            item_factors = new_items

            preds = user_factors.T @ item_factors + self.global_mean
            errs = preds[observed] - matrix[observed]
            self.train_rmse_history.append(float(np.sqrt(np.mean(errs**2))))

        self.user_factors = user_factors
        self.item_factors = item_factors
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        u, i = self._known(user_id, item_id)
        if u is None or i is None:
            return self._clip(self._fallback(user_id, item_id, u, i))
        raw = float(self.user_factors[:, u] @ self.item_factors[:, i] + self.global_mean)
        return self._clip(raw)


class SVDRecommender(_MFBase):
    """Global-mean fill + truncated SVD reconstruction (deterministic)."""

    def __init__(self, n_factors: int = 15):
        super().__init__()
        self.n_factors = n_factors
        self.reconstruction = None

    def fit(self, ratings_df: pd.DataFrame) -> "SVDRecommender":
        matrix = self._setup(ratings_df)
        filled = np.where(np.isnan(matrix), self.global_mean, matrix)
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        k = min(self.n_factors, len(s))
        self.reconstruction = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        u, i = self._known(user_id, item_id)
        if u is None or i is None:
            return self._clip(self._fallback(user_id, item_id, u, i))
        return self._clip(float(self.reconstruction[u, i]))
