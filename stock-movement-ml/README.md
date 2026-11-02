# Stock Movement Prediction

Predicting the **next-day direction** (up/down) of a stock from technical
indicators — with a methodology that survives contact with reality:
**walk-forward validation**, out-of-sample backtesting and transaction costs.

> ⚠️ Educational project. Daily stock direction is close to a coin flip, and
> this project is honest about it — see [Results](#-results). Not financial advice.

## Why walk-forward?

A single chronological 80/20 split evaluates the model in exactly one market
regime. **Walk-forward validation** retrains on an expanding window and tests
on the next unseen slice, rotating through the whole test period — the same
loop a production retraining schedule would use. Combined with
signal-to-next-day-return alignment, it removes the two classic leakage bugs
in financial ML:

- features computed with rolling windows never see the future,
- the signal predicted at the close of day *t* earns the return of day *t+1*.

## Pipeline

```
yfinance (cached CSV) ──► technical indicators (20) ──► models ──► walk-forward (5 folds)
                                                                     │
   reports/dashboard.html ◄── Plotly dashboard ◄── OOS backtest ◄────┘
```

| Module | What it does |
|---|---|
| `src/data_loader.py` | yfinance download + MultiIndex flattening + local CSV cache |
| `src/features.py` | 20 features: returns & lags, SMA/EMA, RSI, MACD, Bollinger, ATR, OBV, volume ratio, volatility |
| `src/train.py` | chronological split, Logistic Regression (scaled) / Random Forest / XGBoost |
| `src/evaluate.py` | accuracy, precision, recall, F1, ROC-AUC comparison table |
| `src/backtest.py` | walk-forward folds + leakage-free long/flat backtest (Sharpe, CAGR, max drawdown, costs) |
| `src/dashboard.py` | single-file Plotly dashboard (price, equity curves, fold metrics, feature importance) |
| `main.py` | end-to-end CLI orchestrating everything into `reports/` |

## Quick start

```bash
pip install -r requirements.txt
python main.py --ticker AAPL --start 2018-01-01 --end 2025-01-01 --n-splits 5
```

Outputs (all written to `reports/`): `model_comparison.csv`,
`walkforward_<model>.csv`, `equity_curve.csv`, `backtest_metrics.json`,
`dashboard.html`.

## 📊 Results — AAPL 2018 → 2025 (1,741 trading days)

**Single chronological 80/20 split (test = last 349 days):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (scaled) | 0.4728 | 0.5393 | 0.4848 | 0.5106 | 0.4356 |
| Random Forest | 0.4241 | 0.4681 | 0.1111 | 0.1796 | 0.4605 |
| XGBoost | 0.4384 | 0.5109 | 0.2374 | 0.3241 | 0.4848 |

**Walk-forward validation** (5 expanding folds, 696 out-of-sample days,
mean over folds):

| Model | Mean Accuracy | Mean ROC-AUC |
|---|---|---|
| Logistic Regression | 0.4952 | 0.4849 |
| Random Forest | 0.4795 | **0.5333** |
| XGBoost | 0.4679 | 0.5128 |

Random Forest per-fold (the honest picture — regimes differ a lot):

| Fold | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| 0 | 0.5396 | 0.5676 | 0.5843 |
| 1 | 0.5324 | 0.5860 | 0.4805 |
| 2 | 0.4604 | 0.1176 | 0.5958 |
| 3 | 0.4820 | 0.4462 | 0.4296 |
| 4 | 0.3830 | 0.0000 | 0.5764 |

**Out-of-sample backtest** (best model = Random Forest, $10k, 0.1% cost per
position change):

| Metric | Strategy | Buy & Hold |
|---|---|---|
| Total return | **+25.45%** | +50.41% |
| CAGR | 8.55% | — |
| Sharpe | 0.52 | — |
| Max drawdown | −16.61% | — |
| Market exposure | 33.3% | 100% |
| Position changes | 142 | 1 |

### What this actually tells you

- Mean walk-forward accuracy ≈ 0.48–0.50 and ROC-AUC ≈ 0.53: daily direction
  is **barely better than a coin flip** — exactly what credible research finds.
- The strategy made half the buy-and-hold return while being invested only a
  third of the time; risk-adjusted (Sharpe, drawdown), the gap is smaller.
- Any model reporting 70–90% accuracy on daily direction is almost certainly
  leaking the future — which is precisely what walk-forward + the
  signal/return alignment in `backtest.py` are designed to expose.

## Tests

```bash
python -m pytest tests/ -q
```

27 tests cover: indicator math (RSI bounds, MACD identity, SMA), target
correctness with **no look-ahead**, fold structure (expanding, contiguous,
non-overlapping), equity-curve arithmetic against hand-computed values,
cost impact, and the cache/flatten helpers of the data loader.
