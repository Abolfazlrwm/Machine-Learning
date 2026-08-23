"""User-based collaborative filtering.

Idea: to predict how much user U will like movie M, find the other
users most similar to U (by rating pattern, via cosine similarity on
mean-centered ratings) and take a similarity-weighted average of
their ratings for M.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    def __init__(self, k_neighbors: int = 30):
        self.k_neighbors = k_neighbors
        self.user_item: pd.DataFrame | None = None
        self.user_means: pd.Series | None = None
        self.centered: pd.DataFrame | None = None
        self.similarity: pd.DataFrame | None = None

    def fit(self, user_item: pd.DataFrame) -> "UserBasedCF":
        self.user_item = user_item
        self.user_means = user_item.mean(axis=1, skipna=True)
        self.centered = user_item.sub(self.user_means, axis=0).fillna(0)

        sim = cosine_similarity(self.centered.values)
        np.fill_diagonal(sim, 0)  # a user is never their own neighbor
        self.similarity = pd.DataFrame(
            sim, index=user_item.index, columns=user_item.index
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        if self.user_item is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.user_item.index:
            return float(self.user_means.mean())  # unknown user: global fallback
        if movie_id not in self.user_item.columns:
            return float(self.user_means.loc[user_id])  # unknown movie: user's own mean

        raters = self.user_item[movie_id].dropna()
        raters = raters.drop(index=user_id, errors="ignore")
        if raters.empty:
            return float(self.user_means.loc[user_id])

        sims = self.similarity.loc[user_id, raters.index]
        top = sims.sort_values(ascending=False).head(self.k_neighbors)
        top = top[top > 0]
        if top.empty:
            return float(self.user_means.loc[user_id])

        deviations = raters.loc[top.index] - self.user_means.loc[top.index]
        weighted = (top.values * deviations.values).sum() / (np.abs(top.values).sum())
        prediction = self.user_means.loc[user_id] + weighted
        return float(np.clip(prediction, 0.5, 5.0))  # stay on the valid rating scale

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-N unrated movies for a user, ranked by predicted rating."""
        if self.user_item is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.user_item.index:
            return []
        unrated = self.user_item.loc[user_id][self.user_item.loc[user_id].isna()].index
        scored = [(int(m), self.predict(user_id, m)) for m in unrated]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
