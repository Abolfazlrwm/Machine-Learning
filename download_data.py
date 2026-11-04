#!/usr/bin/env python3
"""Download the datasets this repo's projects train on.

Datasets live outside git to keep the repository lightweight:

  - recommender-system: MovieLens "ml-latest-small" (GroupLens, CC-BY 4.0)
    -> recommender-system/data/ratings.csv, movies.csv
  - ai-text-detector:   Human vs AI Generated Text (Kaggle)
    -> ai-text-detector/data/human_vs_ai_text.csv
    (mirrored byte-for-byte as a release asset of this repo)

Usage:
    python download_data.py           # fetch whatever is missing
    python download_data.py --force   # re-download everything
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_FILES = ("ratings.csv", "movies.csv")
MOVIELENS_DIR = ROOT / "recommender-system" / "data"

AI_TEXT_URL = ("https://github.com/Abolfazlrwm/Machine-Learning/releases/"
               "download/datasets-v1/human_vs_ai_text.csv")
AI_TEXT_PATH = ROOT / "ai-text-detector" / "data" / "human_vs_ai_text.csv"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "download_data.py"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download project datasets.")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the files already exist")
    args = parser.parse_args()

    missing = [f for f in MOVIELENS_FILES
               if args.force or not (MOVIELENS_DIR / f).exists()]
    if missing:
        print(f"Downloading MovieLens ml-latest-small ({', '.join(missing)}) ...")
        blob = fetch(MOVIELENS_URL)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in missing:
                target = MOVIELENS_DIR / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(f"ml-latest-small/{name}"))
                print(f"  {target.relative_to(ROOT)}  "
                      f"({target.stat().st_size / 1e6:.1f} MB)")
    else:
        print("MovieLens files already present.")

    if args.force or not AI_TEXT_PATH.exists():
        print("Downloading human_vs_ai_text.csv (release asset) ...")
        AI_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        AI_TEXT_PATH.write_bytes(fetch(AI_TEXT_URL))
        print(f"  {AI_TEXT_PATH.relative_to(ROOT)}  "
              f"({AI_TEXT_PATH.stat().st_size / 1e6:.1f} MB)")
    else:
        print("human_vs_ai_text.csv already present.")

    print("Done.")


if __name__ == "__main__":
    main()
