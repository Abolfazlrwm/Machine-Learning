"""Dataset loading, the sklearn pipeline and the GridSearchCV training
entry point. `train_model` returns its metrics so tests and callers never
have to parse stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "spam_classifier.pkl"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Read the UCI CSV (latin-1, `v1`/`v2` columns) into label/message rows."""
    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    if df["label"].isna().any():
        raise ValueError("dataset contains labels other than ham/spam")
    df["cleaned"] = df["message"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF (max 5,000 features, unigrams + bigrams) → LogisticRegression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression()),
    ])


PARAM_GRID: dict[str, list] = {"clf__C": [0.1, 1, 10], "clf__solver": ["liblinear"]}


def train_model(csv_path: str | Path, model_out: str | Path = DEFAULT_MODEL_PATH,
                seed: int = 42) -> dict:
    """Full training run: split → 5-fold grid search → holdout metrics →
    5-fold CV accuracy → persist the best pipeline. Returns the metrics."""
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.2, random_state=seed)

    grid = GridSearchCV(build_pipeline(), PARAM_GRID, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "best_params": grid.best_params_,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])),
        "cv_accuracy": float(cross_val_score(best, df["cleaned"], df["label"], cv=5).mean()),
        "spam_precision": float(report["1"]["precision"]),
        "spam_recall": float(report["1"]["recall"]),
        "spam_f1": float(report["1"]["f1-score"]),
    }

    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to spam.csv")
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    metrics = train_model(args.train, args.model_out)
    print(f"Best parameters : {metrics['best_params']}")
    print(f"Accuracy (holdout)      : {metrics['accuracy']:.4f}")
    print(f"ROC-AUC (holdout)       : {metrics['roc_auc']:.4f}")
    print(f"CV accuracy (5-fold)    : {metrics['cv_accuracy']:.4f}")
    print(f"Spam precision / recall / f1 : "
          f"{metrics['spam_precision']:.2f} / {metrics['spam_recall']:.2f} / {metrics['spam_f1']:.2f}")
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
