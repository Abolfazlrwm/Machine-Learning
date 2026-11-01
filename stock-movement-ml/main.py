"""End-to-end pipeline: data -> features -> models -> walk-forward -> backtest.

Examples:
    python main.py --ticker AAPL --start 2018-01-01 --end 2025-01-01
    python main.py --ticker MSFT --n-splits 6 --refresh
    python main.py --no-dashboard --no-llm-free
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.backtest import backtest_strategy, walk_forward_validation
from src.dashboard import make_dashboard
from src.data_loader import load_data
from src.evaluate import evaluate_models, get_feature_importance
from src.features import calculate_technical_indicators, create_target
from src.train import split_data, train_models

REPORTS_DIR = Path("reports")

# Fresh, unfitted models per walk-forward fold. Logistic Regression uses the
# same scaled pipeline as src/train.py so all experiments stay comparable.
MODEL_FACTORIES = {
    "Logistic_Regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Random_Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": lambda: XGBClassifier(eval_metric="logloss", random_state=42),
}


def run(args: argparse.Namespace) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Data ------------------------------------------------------------
    print(f"[1/5] Loading {args.ticker} {args.start} -> {args.end} ...")
    raw = load_data(args.ticker, args.start, args.end, refresh=args.refresh)
    print(f"      {len(raw)} trading days cached in data/")

    # 2. Features ----------------------------------------------------------
    print("[2/5] Building technical indicators and target ...")
    df = create_target(calculate_technical_indicators(raw))
    print(f"      usable rows after warm-up: {len(df)}")

    # 3. Single-holdout model comparison -----------------------------------
    print("[3/5] Training models on a chronological 80/20 split ...")
    X_train, X_test, y_train, y_test = split_data(df, train_size=args.train_size)
    models = train_models(X_train, y_train)
    comparison = evaluate_models(models, X_test, y_test)
    print(comparison.round(4).to_string(), "\n")
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv")

    # 4. Walk-forward validation -------------------------------------------
    print(f"[4/5] Walk-forward validation ({args.n_splits} folds, "
          f"min_train={args.min_train_frac:.0%}) ...")
    folds_tables: dict[str, pd.DataFrame] = {}
    oos_tables: dict[str, pd.DataFrame] = {}
    for name, factory in MODEL_FACTORIES.items():
        folds_df, oos_df = walk_forward_validation(
            df, factory, model_name=name,
            n_splits=args.n_splits, min_train_frac=args.min_train_frac,
        )
        folds_tables[name] = folds_df
        oos_tables[name] = oos_df
        folds_df.round(4).to_csv(REPORTS_DIR / f"walkforward_{name}.csv")
        print(f"      {name:<20} mean accuracy={folds_df.loc['mean', 'Accuracy']:.4f} "
              f"mean ROC-AUC={folds_df.loc['mean', 'ROC-AUC']:.4f}")

    best_model = max(
        folds_tables, key=lambda n: folds_tables[n].loc["mean", "ROC-AUC"]
    )
    print(f"      -> best model by mean ROC-AUC: {best_model}\n")

    # 5. Out-of-sample backtest + dashboard --------------------------------
    print("[5/5] Backtesting out-of-sample signals ...")
    oos_df = oos_tables[best_model]
    equity, metrics = backtest_strategy(
        df.loc[oos_df.index], oos_df["OOS_Signal"],
        initial_capital=args.capital, transaction_cost=args.cost,
    )
    metrics["model"] = best_model
    print(json.dumps(metrics, indent=2))
    (REPORTS_DIR / "backtest_metrics.json").write_text(json.dumps(metrics, indent=2))
    equity.to_csv(REPORTS_DIR / "equity_curve.csv")

    if not args.no_dashboard:
        importance = get_feature_importance(
            models[best_model], list(X_train.columns)
        )
        make_dashboard(
            price_df=df, equity_df=equity, folds_df=folds_tables[best_model],
            feature_importance=importance, ticker=args.ticker,
            out_path=REPORTS_DIR / "dashboard.html",
        )
        print(f"Dashboard saved to {REPORTS_DIR / 'dashboard.html'}")

    print("\nDone. Reports are in ./reports/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock movement prediction pipeline")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-frac", type=float, default=0.6)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--cost", type=float, default=0.001,
                        help="proportional transaction cost per position change")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download data even if cached")
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
