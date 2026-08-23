"""Load and prepare the MovieLens ml-latest-small dataset."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load ratings.csv -> DataFrame[userId, movieId, rating, timestamp]."""
    return pd.read_csv(data_dir / "ratings.csv")


def load_movies(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load movies.csv -> DataFrame[movieId, title, genres]."""
    return pd.read_csv(data_dir / "movies.csv")


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot ratings into a dense user x movie matrix (NaN = not rated)."""
    return ratings.pivot(index="userId", columns="movieId", values="rating")


def train_test_split_by_user(
    ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `test_size` of each user's ratings for evaluation.

    A plain random split can leave some users entirely out of the
    training set, which breaks collaborative filtering. Splitting
    per-user avoids that.
    """
    test_rows = (
        ratings.groupby("userId", group_keys=False)
        .apply(lambda g: g.sample(frac=test_size, random_state=random_state), include_groups=False)
    )
    test_rows["userId"] = ratings.loc[test_rows.index, "userId"]
    train = ratings.drop(test_rows.index)
    return train, test_rows
