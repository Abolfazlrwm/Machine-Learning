"""
base_recommender.py
--------------------
Abstract interface every recommender in this project implements.

Any new domain (books, music, ...) added in later phases only needs
to subclass BaseRecommender and implement fit/predict/recommend —
the evaluation and demo layers stay identical across domains.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Common interface for all recommender implementations."""

    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame) -> "BaseRecommender":
        """Train the model on a ratings DataFrame with columns
        [user_id, item_id, rating]. Returns self for chaining."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating a user would give an item."""
        raise NotImplementedError

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list:
        """Return the top-N recommended item_ids for a user,
        excluding items the user has already rated."""
        raise NotImplementedError

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Compute RMSE and MAE on a held-out test set.

        Shared across all subclasses — no need to reimplement per domain.
        """
        errors = []
        for _, row in test_df.iterrows():
            pred = self.predict(row.user_id, row.item_id)
            if pred is not None and not np.isnan(pred):
                errors.append(pred - row.rating)

        errors = np.array(errors)
        if len(errors) == 0:
            return {"rmse": float("nan"), "mae": float("nan"), "n_evaluated": 0}

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        return {"rmse": rmse, "mae": mae, "n_evaluated": len(errors)}

    def precision_recall_at_k(self, test_df: pd.DataFrame, k: int = 10,
                               threshold: float = 4.0) -> dict:
        """Precision@K / Recall@K: of the top-K items we recommend,
        how many does the user actually like (rating >= threshold)?"""
        precisions, recalls = [], []

        for user_id, group in test_df.groupby("user_id"):
            relevant_items = set(group[group.rating >= threshold].item_id)
            if not relevant_items:
                continue

            recommended = set(self.recommend(user_id, n=k))
            n_relevant_and_recommended = len(relevant_items & recommended)

            precisions.append(n_relevant_and_recommended / k)
            recalls.append(n_relevant_and_recommended / len(relevant_items))

        return {
            "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
            "recall_at_k": float(np.mean(recalls)) if recalls else float("nan"),
            "k": k,
        }
