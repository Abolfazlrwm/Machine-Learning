import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from .features import FEATURE_COLUMNS
except ImportError:  # allows running this file directly: python src/train.py
    from features import FEATURE_COLUMNS

import joblib
from pathlib import Path


def split_data(df: pd.DataFrame, train_size: float = 0.8):
    """
    Splits the data into training and testing sets based on time.
    No shuffling to preserve the temporal order of stock prices.
    """
    X = df[FEATURE_COLUMNS]
    y = df['Target']

    # Calculate split index
    split_idx = int(len(df) * train_size)

    # Time-based split
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Trains Logistic Regression, Random Forest, and XGBoost models.
    Returns a dictionary of trained models.
    """
    models = {}

    # 1. Logistic Regression (Baseline) — scaled, because features live on
    #    very different ranges (RSI 0-100, returns ~0.01, OBV huge)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_train, y_train)
    models['Logistic_Regression'] = lr

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random_Forest'] = rf

    # 3. XGBoost
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb

    print("All models trained successfully!")
    return models


def persist_models(models: dict, ticker: str, out_dir: str | Path = "models") -> list:
    """Save fitted models as ``{out_dir}/{ticker}_{name}.joblib`` for serving.

    The FastAPI layer (``serving/``) loads these artifacts to answer
    ``GET /v1/stock/signal`` without retraining. Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in models.items():
        path = out_dir / f"{ticker.upper()}_{name}.joblib"
        joblib.dump(model, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    # This block is for independent testing of this module
    try:
        data = pd.read_csv("data/AAPL_processed.csv")
        X_train, X_test, y_train, y_test = split_data(data)
        trained_models = train_models(X_train, y_train)
    except FileNotFoundError:
        print("Processed data not found. Please run the full pipeline.")
