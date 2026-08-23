"""RMSE / MAE evaluation for a fitted recommender against a held-out
test set of (userId, movieId, rating) rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(model, test_ratings: pd.DataFrame) -> dict[str, float]:
    errors = []
    for _, row in test_ratings.iterrows():
        pred = model.predict(int(row.userId), int(row.movieId))
        errors.append(pred - row.rating)

    errors = np.array(errors)
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    return {"rmse": rmse, "mae": mae, "n": len(errors)}
