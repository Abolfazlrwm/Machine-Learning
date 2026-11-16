"""Load and prepare the MovieLens ml-latest-small dataset.

The dataset is kept out of git (root ``.gitignore``). Restore it with
``python -m src.data_loader`` (or ``load_ratings(download=True)``) which
fetches the official zip from grouplens.org and extracts the two CSVs.

All frames use the repo-wide ``user_id``/``item_id`` column names the
recommender interfaces expect — the loader renames MovieLens'
``userId``/``movieId`` once, at the boundary.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def _ensure_data(download: bool, data_dir: Path) -> None:
    """Raise FileNotFoundError (or download) when the CSVs are missing."""
    if (data_dir / "ratings.csv").exists():
        return
    if not download:
        raise FileNotFoundError(
            f"MovieLens data not found in {data_dir} — run "
            "`python -m src.data_loader` (or pass download=True) first."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(MOVIELENS_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for member in ("ml-latest-small/ratings.csv", "ml-latest-small/movies.csv"):
        (data_dir / Path(member).name).write_bytes(archive.read(member))
    print(f"downloaded MovieLens ml-latest-small -> {data_dir}")


def load_ratings(data_dir: Path = DATA_DIR, download: bool = False) -> pd.DataFrame:
    """Load ratings.csv -> DataFrame[user_id, item_id, rating, timestamp]."""
    _ensure_data(download, data_dir)
    frame = pd.read_csv(data_dir / "ratings.csv")
    return frame.rename(columns={"userId": "user_id", "movieId": "item_id"})


def load_movies(data_dir: Path = DATA_DIR, download: bool = False) -> pd.DataFrame:
    """Load movies.csv -> DataFrame[item_id, title, genres]."""
    _ensure_data(download, data_dir)
    frame = pd.read_csv(data_dir / "movies.csv")
    return frame.rename(columns={"movieId": "item_id"})


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot ratings into a dense user x item matrix (NaN = not rated)."""
    return ratings.pivot(index="user_id", columns="item_id", values="rating")


def train_test_split_by_user(
    ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `test_size` of each user's ratings for evaluation.

    A plain random split can leave some users entirely out of the
    training set, which breaks collaborative filtering. Splitting
    per-user avoids that.
    """
    test_rows = (
        ratings.groupby("user_id", group_keys=False)
        .apply(lambda g: g.sample(frac=test_size, random_state=random_state), include_groups=False)
    )
    test_rows["user_id"] = ratings.loc[test_rows.index, "user_id"]
    train = ratings.drop(test_rows.index)
    return train, test_rows


if __name__ == "__main__":
    load_ratings(download=True)
    load_movies(download=True)
