"""
data_loader.py
---------------
Downloads (if needed) and loads the MovieLens 100K dataset into a
clean pandas DataFrame with columns: user_id, item_id, rating, timestamp.

MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
"""

import os
import zipfile
import urllib.request

import pandas as pd

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_movielens(data_dir: str = DATA_DIR, force: bool = False) -> str:
    """Download and extract the MovieLens 100K dataset.

    Returns the path to the extracted ``ml-100k`` folder.
    """
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, "ml-100k")
    zip_path = os.path.join(data_dir, "ml-100k.zip")

    if os.path.isdir(extracted_dir) and not force:
        return extracted_dir

    print(f"Downloading MovieLens 100K from {ML_100K_URL} ...")
    urllib.request.urlretrieve(ML_100K_URL, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    return extracted_dir


def load_ratings(data_dir: str = DATA_DIR, download: bool = True) -> pd.DataFrame:
    """Load the ratings table as a DataFrame.

    Columns: user_id (int), item_id (int), rating (int 1-5), timestamp (int)
    """
    extracted_dir = os.path.join(data_dir, "ml-100k")
    ratings_path = os.path.join(extracted_dir, "u.data")

    if not os.path.exists(ratings_path):
        if not download:
            raise FileNotFoundError(
                f"Ratings file not found at {ratings_path}. "
                "Run download_movielens() first or pass download=True."
            )
        download_movielens(data_dir)

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    return ratings


def load_items(data_dir: str = DATA_DIR, download: bool = True) -> pd.DataFrame:
    """Load the movie metadata table (id, title, genres, ...)."""
    extracted_dir = os.path.join(data_dir, "ml-100k")
    items_path = os.path.join(extracted_dir, "u.item")

    if not os.path.exists(items_path):
        if not download:
            raise FileNotFoundError(f"Items file not found at {items_path}.")
        download_movielens(data_dir)

    columns = [
        "item_id", "title", "release_date", "video_release_date", "imdb_url",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    items = pd.read_csv(
        items_path, sep="|", names=columns, encoding="latin-1"
    )
    return items


if __name__ == "__main__":
    df = load_ratings()
    print(df.head())
    print(f"\n{len(df)} ratings from {df.user_id.nunique()} users "
          f"on {df.item_id.nunique()} movies")
