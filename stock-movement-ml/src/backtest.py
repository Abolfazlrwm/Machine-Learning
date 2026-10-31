"""Walk-forward validation and strategy backtesting.

Why walk-forward?
-----------------
A single chronological 80/20 split evaluates the model in exactly *one*
market regime. Walk-forward validation retrains on an expanding window
and tests on the next unseen slice, rotating through the whole test
period — much closer to how the model would be used in production.

The backtest is leakage-free by construction: the signal for day *t* is
predicted from features known at the close of day *t*, and earns the
return of day *t+1* (``Close[t+1] / Close[t] - 1``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .features import FEATURE_COLUMNS
    from .evaluate import evaluate_models, get_feature_importance
except ImportError:  # allows running this file directly
    from features import FEATURE_COLUMNS
    from evaluate import evaluate_models, get_feature_importance


# --------------------------------------------------------------------- folds
@dataclass
class Fold:
    index: int
    train_slice: slice
    test_slice: slice


def walk_forward_splits(n_samples: int, n_splits: int = 5,
                        min_train_frac: float = 0.6) -> list[Fold]:
    """Expanding-window folds: train [0 .. t), test [t .. t+step).

    The last ``1 - min_train_frac`` of samples is carved into ``n_splits``
    equal test slices. Every fold trains on *all* data before its test slice.
    """
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must be in (0, 1)")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    train_min = int(n_samples * min_train_frac)
    test_size = (n_samples - train_min) // n_splits
    if test_size < 1:
        raise ValueError("Not enough samples for this n_splits/min_train_frac")

    folds = []
    for i in range(n_splits):
        train_end = train_min + i * test_size
        test_end = train_end + test_size if i < n_splits - 1 else n_samples
        folds.append(Fold(index=i, train_slice=slice(0, train_end),
                          test_slice=slice(train_end, test_end)))
    return folds


# --------------------------------------------------------------- validation
def walk_forward_validation(
    df: pd.DataFrame,
    model_factory,
    model_name: str = "model",
    n_splits: int = 5,
    min_train_frac: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window walk-forward for one model.

    Parameters
    ----------
    df : feature/target frame (as produced by features.create_target)
    model_factory : callable returning a *fresh, unfitted* model
    model_name : label used in the results table

    Returns
    -------
    folds_df : per-fold metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
    oos_df : out-of-sample predictions for every test row
             (columns: OOS_Signal, OOS_Proba, Target)
    """
    folds = walk_forward_splits(len(df), n_splits=n_splits, min_train_frac=min_train_frac)
    X = df[FEATURE_COLUMNS]
    y = df['Target']

    fold_frames: list[pd.DataFrame] = []
    oos_parts: list[pd.DataFrame] = []

    for fold in folds:
        X_train, y_train = X.iloc[fold.train_slice], y.iloc[fold.train_slice]
        X_test, y_test = X.iloc[fold.test_slice], y.iloc[fold.test_slice]

        model = model_factory()
        model.fit(X_train, y_train)
        metrics = evaluate_models({model_name: model}, X_test, y_test)
        metrics.insert(0, "Fold", fold.index)
        fold_frames.append(metrics)

        proba = (model.predict_proba(X_test)[:, 1]
                 if hasattr(model, "predict_proba") else np.full(len(X_test), np.nan))
        oos_parts.append(pd.DataFrame({
            "OOS_Signal": model.predict(X_test),
            "OOS_Proba": proba,
            "Target": y_test.values,
        }, index=X_test.index))

    folds_df = pd.concat(fold_frames).set_index("Fold")
    folds_df.loc["mean"] = folds_df.mean(numeric_only=True)
    oos_df = pd.concat(oos_parts)
    return folds_df, oos_df


# ------------------------------------------------------------------ backtest
def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """Simulate a long/flat strategy from per-day signals.

    Parameters
    ----------
    df : frame with a Close column (signals index must align with df)
    signals : 0/1 position decided at the close of day *t*
    initial_capital : starting equity
    transaction_cost : proportional cost paid on position changes

    Returns
    -------
    equity_df : date-indexed frame with Strategy and BuyHold equity curves
    metrics : total/CAGR return, Sharpe, max drawdown, exposure, trades
    """
    if not signals.index.equals(df.index):
        raise ValueError("signals index must match df index")

    next_ret = df['Close'].pct_change().shift(-1)   # return earned from t to t+1
    position = signals.astype(float)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * transaction_cost
    strat_ret = position * next_ret - cost

    valid = next_ret.notna()
    strat_ret, next_ret = strat_ret[valid], next_ret[valid]

    equity = pd.DataFrame({
        "Strategy": initial_capital * (1.0 + strat_ret).cumprod(),
        "BuyHold": initial_capital * (1.0 + next_ret).cumprod(),
    })

    n_days = len(strat_ret)
    total_return = equity["Strategy"].iloc[-1] / initial_capital - 1.0
    bh_return = equity["BuyHold"].iloc[-1] / initial_capital - 1.0
    years = max(n_days / 252.0, 1e-9)
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    std = strat_ret.std()
    sharpe = (strat_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    running_max = equity["Strategy"].cummax()
    max_dd = (equity["Strategy"] / running_max - 1.0).min()

    direction = (df['Close'].shift(-1) > df['Close']).astype(int)[valid]
    hit_rate = float((signals[valid] == direction).mean())

    metrics = {
        "days": int(n_days),
        "total_return": round(float(total_return), 4),
        "buyhold_return": round(float(bh_return), 4),
        "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "exposure": round(float(position.mean()), 4),
        "n_trades": int((turnover > 0).sum()),
        "directional_accuracy": round(hit_rate, 4),
    }
    return equity, metrics
