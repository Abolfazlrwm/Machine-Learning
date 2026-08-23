"""Item-based collaborative filtering.

Idea: to predict how much user U will like movie M, find movies most
similar to M *based on how the whole userbase rated them* (cosine
similarity on mean-centered item vectors), then take a
similarity-weighted average of U's own ratings for those movies.

Item-item similarity is far more stable than user-user similarity for
this kind of dataset (movies get re-rated by new users constantly,
users' own taste is comparatively stable), and it's the same
underlying idea Amazon's "customers who bought this also bought"
popularized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    def __init__(self, k_neighbors: int = 30):
        self.k_neighbors = k_neighbors
        self.user_item: pd.DataFrame | None = None
        self.item_means: pd.Series | None = None
        self.similarity: pd.DataFrame | None = None

    def fit(self, user_item: pd.DataFrame) -> "ItemBasedCF":
        self.user_item = user_item
        self.item_means = user_item.mean(axis=0, skipna=True)
        centered = user_item.sub(self.item_means, axis=1).fillna(0)

        sim = cosine_similarity(centered.values.T)
        np.fill_diagonal(sim, 0)
        self.similarity = pd.DataFrame(
            sim, index=user_item.columns, columns=user_item.columns
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if self.user_item is None:
            raise RuntimeError("Call fit() first.")
        if movie_id not in self.user_item.columns:
            return float(self.item_means.mean())
        if user_id not in self.user_item.index:
            return float(self.item_means.loc[movie_id])

        user_ratings = self.user_item.loc[user_id].dropna()
        user_ratings = user_ratings.drop(index=movie_id, errors="ignore")
        if user_ratings.empty:
            return float(self.item_means.loc[movie_id])

        sims = self.similarity.loc[movie_id, user_ratings.index]
        top = sims.sort_values(ascending=False).head(self.k_neighbors)
        top = top[top > 0]
        if top.empty:
            return float(self.item_means.loc[movie_id])

        weighted = (top.values * user_ratings.loc[top.index].values).sum() / np.abs(top.values).sum()
        return float(np.clip(weighted, 0.5, 5.0))  # stay on the valid rating scale

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        if self.user_item is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.user_item.index:
            return []
        unrated = self.user_item.loc[user_id][self.user_item.loc[user_id].isna()].index
        scored = [(int(m), self.predict(user_id, m)) for m in unrated]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
