# Stock movement ML

`stock-movement-ml/` predicts the **next-day direction** (up/down) of a
stock from 20 technical indicators — with a methodology built to survive
contact with reality: **walk-forward validation**, out-of-sample
backtesting and transaction costs. The honesty *is* the feature: most
tutorial projects hide the fact that directional prediction hovers around
a coin flip; this one measures it and shows it.

## Pipeline

```
yfinance (cached CSV) ──► 20 technical indicators ──► models ──► walk-forward (5 expanding folds)
                                                                        │
              dashboard ◄── equity curves ◄── leakage-free long/flat backtest (costs)
```

| module | role |
|---|---|
| `src/train.py` | chronological split; Logistic Regression (scaled) / Random Forest / XGBoost |
| `src/evaluate.py` | accuracy, precision, recall, F1, ROC-AUC comparison |
| `src/backtest.py` | walk-forward folds + leakage-free long/flat backtest (Sharpe, CAGR, max drawdown, costs) |
| `src/dashboard.py` | single-file Plotly dashboard: price, equity curves, fold metrics, feature importance |

## Why walk-forward

A single chronological 80/20 split evaluates the model in exactly one
market regime. Five expanding folds re-train on history and test on the
future, so the reported score is a distribution over regimes — and the
per-fold table shows how much it moves.

## Results — AAPL 2018 → 2025 (1,741 trading days)

Single split (last 349 days) vs walk-forward mean over 696 out-of-sample
days:

| model | single-split accuracy | walk-forward mean accuracy | walk-forward mean ROC-AUC |
|---|---|---|---|
| Logistic Regression (scaled) | 0.4728 | 0.4952 | 0.4849 |
| Random Forest | 0.4241 | 0.4795 | **0.5333** |
| XGBoost | 0.4384 | 0.4679 | 0.5128 |

Read it like an adult: an ROC-AUC of 0.53 is a *thin* edge that costs,
slippage and regime changes can erase — which is exactly what the
long/flat backtest (with transaction costs) demonstrates. The lesson the
repo teaches on purpose: **validate the way you would trade, not the way
that flatters the model.**

All artifacts land in `reports/` (`model_comparison.csv`,
`walkforward_<model>.csv`, `equity_curve.csv`, `backtest_metrics.json`)
and the Plotly dashboard renders them in one file.
