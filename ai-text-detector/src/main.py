import time
import joblib
from pathlib import Path

from data_loader import load_dataset, split_dataset
from model import tfidf_pipeline, hybrid_pipeline, MODELS
from evaluate import evaluate, print_report

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


def main():
    print("loading dataset...")
    df = load_dataset()
    print(f"{len(df)} samples total ({(df.label==1).sum()} ai, {(df.label==0).sum()} human)")

    X_train, X_test, y_train, y_test = split_dataset(df)
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print()

    best_score = 0
    best_model = None
    best_name = ""

    for name, make_model in MODELS.items():
        for variant in ["tfidf", "hybrid"]:
            full_name = f"{name} ({variant})"

            start = time.time()
            if variant == "tfidf":
                pipe = tfidf_pipeline(make_model())
            else:
                pipe = hybrid_pipeline(make_model())

            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)
            took = time.time() - start

            print_report(full_name, metrics)
            print(f"  took {took:.1f}s")
            print()

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = pipe
                best_name = full_name

    print(f"best: {best_name} with f1={best_score:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
