"""Inference against the persisted pipeline: clean → predict → label."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.preprocess import clean_text
from src.train import DEFAULT_MODEL_PATH


def predict_message(message: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    """Return 'SPAM' or 'HAM (Not Spam)' for one raw message."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"model not found at {model_path} — train first: "
            f"python -m src.train --train data/spam.csv")
    model = joblib.load(model_path)
    prediction = model.predict([clean_text(message)])[0]
    return "SPAM" if prediction == 1 else "HAM (Not Spam)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", required=True, help="A single SMS message")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    try:
        print(predict_message(args.predict, args.model))
    except FileNotFoundError as err:
        print(err)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
