import sys
import numpy as np
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


def predict(text):
    model = joblib.load(MODEL_PATH)
    pred = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        conf = model.predict_proba([text])[0][pred]
    else:
        # SGDClassifier only gives us decision_function, not a real probability
        # squashing it with sigmoid just so we have something to show
        score = model.decision_function([text])[0]
        conf = 1 / (1 + np.exp(-abs(score)))

    label = "AI-generated" if pred == 1 else "Human-written"
    return label, conf


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print("no model found, run main.py first to train one")
        sys.exit(1)

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("didn't get any text")
        sys.exit(1)

    label, conf = predict(text)
    print(f"{label} (confidence: {conf:.1%})")
