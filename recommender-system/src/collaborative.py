"""
collaborative.py
------------------
User-based and item-based Collaborative Filtering, built on top of a
sparse user-item matrix for scalability.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.base_recommender import BaseRecommender


class _CFBase(BaseRecommender):
    """Shared machinery for user-based and item-based CF: builds the
    sparse user-item matrix and id <-> index lookups."""

    def __init__(self, k_neighbors: int = 30):
        self.k_neighbors = k_neighbors
        self.user_ids = None
        self.item_ids = None
        self.user_index = {}
        self.item_index = {}
        self.matrix = None          # sparse user x item matrix
        self.user_means = None
        self.similarity = None      # computed by subclass
        self._rated_items_by_user = {}

    def fit(self, ratings_df: pd.DataFrame) -> "_CFBase":
        self.user_ids = np.sort(ratings_df.user_id.unique())
        self.item_ids = np.sort(ratings_df.item_id.unique())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}

        rows = ratings_df.user_id.map(self.user_index)
        cols = ratings_df.item_id.map(self.item_index)
        self.matrix = csr_matrix(
            (ratings_df.rating, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )

        dense_row_sums = np.asarray(self.matrix.sum(axis=1)).flatten().astype(float)
        counts = np.diff(self.matrix.tocsr().indptr)
        self.user_means = np.divide(
            dense_row_sums, counts,
            out=np.zeros_like(dense_row_sums, dtype=float),
            where=counts != 0,
        )

        self._rated_items_by_user = {
            u: set(ratings_df[ratings_df.user_id == u].item_id)
            for u in self.user_ids
        }

        self._compute_similarity()
        return self

    def _compute_similarity(self):
        raise NotImplementedError

    def _mean_rating_for_user(self, user_id: int) -> float:
        idx = self.user_index.get(user_id)
        if idx is None:
            return float(np.mean(self.user_means))
        return float(self.user_means[idx])


class UserBasedCF(_CFBase):
    """Predicts ratings using similarity between users."""

    def _compute_similarity(self):
        self.similarity = cosine_similarity(self.matrix)
        np.fill_diagonal(self.similarity, 0)

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_index or item_id not in self.item_index:
            return self._mean_rating_for_user(user_id)

        u_idx = self.user_index[user_id]
        i_idx = self.item_index[item_id]

        sims = self.similarity[u_idx]
        item_ratings = np.asarray(self.matrix[:, i_idx].todense()).flatten()
        rated_mask = item_ratings > 0

        if not rated_mask.any():
            return self._mean_rating_for_user(user_id)

        neighbor_sims = sims[rated_mask]
        top_k = np.argsort(neighbor_sims)[-self.k_neighbors:]

        weights = neighbor_sims[top_k]
        ratings = item_ratings[rated_mask][top_k]

        if weights.sum() == 0:
            return self._mean_rating_for_user(user_id)

        return float(np.dot(weights, ratings) / np.sum(np.abs(weights)))

    def recommend(self, user_id: int, n: int = 10) -> list:
        if user_id not in self.user_index:
            return []
        rated = self._rated_items_by_user.get(user_id, set())
        candidates = [it for it in self.item_ids if it not in rated]
        scored = [(it, self.predict(user_id, it)) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:n]]


class ItemBasedCF(_CFBase):
    """Predicts ratings using similarity between items — generally
    more stable than user-based CF since item similarity changes
    less often than a user's own taste profile."""

    def _compute_similarity(self):
        self.similarity = cosine_similarity(self.matrix.T)
        np.fill_diagonal(self.similarity, 0)

    def predict(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_index or item_id not in self.item_index:
            return self._mean_rating_for_user(user_id)

        u_idx = self.user_index[user_id]
        i_idx = self.item_index[item_id]

        user_ratings = np.asarray(self.matrix[u_idx, :].todense()).flatten()
        rated_mask = user_ratings > 0

        if not rated_mask.any():
            return self._mean_rating_for_user(user_id)

        item_sims = self.similarity[i_idx][rated_mask]
        top_k = np.argsort(item_sims)[-self.k_neighbors:]

        weights = item_sims[top_k]
        ratings = user_ratings[rated_mask][top_k]

        if weights.sum() == 0:
            return self._mean_rating_for_user(user_id)

        return float(np.dot(weights, ratings) / np.sum(np.abs(weights)))

    def recommend(self, user_id: int, n: int = 10) -> list:
        if user_id not in self.user_index:
            return []
        rated = self._rated_items_by_user.get(user_id, set())
        candidates = [it for it in self.item_ids if it not in rated]
        scored = [(it, self.predict(user_id, it)) for it in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:n]]
