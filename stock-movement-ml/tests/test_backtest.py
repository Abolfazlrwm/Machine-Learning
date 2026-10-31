import numpy as np
import pandas as pd

from src.backtest import (
    backtest_strategy,
    walk_forward_splits,
    walk_forward_validation,
)


class TestWalkForwardSplits:
    def test_folds_cover_tail_exactly_once(self):
        folds = walk_forward_splits(100, n_splits=4, min_train_frac=0.6)
        assert len(folds) == 4
        covered = []
        for f in folds:
            covered.extend(range(f.test_slice.start, f.test_slice.stop))
        assert covered == list(range(60, 100))  # tail partitioned, no gaps/overlaps

    def test_train_is_expanding(self):
        folds = walk_forward_splits(300, n_splits=5)
        ends = [f.train_slice.stop for f in folds]
        assert ends == sorted(ends) and len(set(ends)) == len(ends)

    def test_train_never_overlaps_test(self):
        folds = walk_forward_splits(250, n_splits=5)
        for f in folds:
            assert f.train_slice.stop <= f.test_slice.start

    def test_invalid_args_raise(self):
        import pytest

        with pytest.raises(ValueError):
            walk_forward_splits(100, min_train_frac=1.5)
        with pytest.raises(ValueError):
            walk_forward_splits(10, n_splits=20)  # too few samples


class TestWalkForwardValidation:
    def _factory(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000)

    def test_per_fold_metrics_and_mean_row(self, features_df):
        folds_df, oos_df = walk_forward_validation(
            features_df, self._factory, model_name="lr",
            n_splits=4, min_train_frac=0.6,
        )
        assert list(folds_df.index) == [0, 1, 2, 3, "mean"]
        assert ((folds_df["Accuracy"] >= 0) & (folds_df["Accuracy"] <= 1)).all()
        assert len(oos_df) == sum(
            f.test_slice.stop - f.test_slice.start for f in
            walk_forward_splits(len(features_df), 4, 0.6)
        )
        assert set(oos_df.columns) == {"OOS_Signal", "OOS_Proba", "Target"}

    def test_oos_predictions_align_with_test_indices(self, features_df):
        _, oos_df = walk_forward_validation(
            features_df, self._factory, n_splits=3, min_train_frac=0.7,
        )
        assert oos_df.index.isin(features_df.index).all()


class TestBacktestStrategy:
    def _tiny_frame(self):
        closes = [100.0, 110.0, 99.0, 105.0]
        idx = pd.bdate_range("2024-01-01", periods=4)
        return pd.DataFrame({"Close": closes}, index=idx)

    def test_manual_equity_math_no_cost(self):
        df = self._tiny_frame()
        signals = pd.Series([1, 1, 0], index=df.index[:-1]).reindex(df.index).ffill()
        signals = pd.Series([1, 1, 0, np.nan], index=df.index)  # t=0,1,2 known
        signals = signals.fillna(0)

        equity, metrics = backtest_strategy(df, signals, initial_capital=1000.0,
                                            transaction_cost=0.0)
        # next returns: +10%, -10%, +6.06% ; valid rows drop the last (NaN)
        # strategy returns: 1*0.10, 1*(-0.10), 0*0.0606
        expected = 1000.0 * 1.10 * 0.90
        np.testing.assert_allclose(equity["Strategy"].iloc[-1], expected, rtol=1e-9)

    def test_buyhold_curve_matches_cumprod(self):
        df = self._tiny_frame()
        signals = pd.Series(1.0, index=df.index)
        equity, _ = backtest_strategy(df, signals, initial_capital=1000.0,
                                      transaction_cost=0.0)
        rets = df["Close"].pct_change().shift(-1).dropna()
        expected = 1000.0 * (1 + rets).cumprod()
        np.testing.assert_allclose(equity["BuyHold"].values, expected.values, rtol=1e-9)

    def test_costs_reduce_returns_on_flip(self):
        df = self._tiny_frame()
        flip = pd.Series([1, 0, 1, 0], index=df.index, dtype=float)  # flips every day
        _, m_no_cost = backtest_strategy(df, flip, transaction_cost=0.0)
        _, m_cost = backtest_strategy(df, flip, transaction_cost=0.01)
        assert m_cost["total_return"] < m_no_cost["total_return"], \
            "identical signals must earn less once costs are applied"

    def test_index_mismatch_raises(self):
        import pytest

        df = self._tiny_frame()
        bad = pd.Series([1, 0], index=df.index[:2])
        with pytest.raises(ValueError):
            backtest_strategy(df, bad)

    def test_metrics_keys(self, features_df):
        signals = pd.Series(1.0, index=features_df.index)
        _, metrics = backtest_strategy(features_df, signals, transaction_cost=0.0)
        for key in ("total_return", "buyhold_return", "sharpe", "max_drawdown",
                    "directional_accuracy", "n_trades", "exposure"):
            assert key in metrics
        # always-in-market strategy == buy & hold modulo rounding
        np.testing.assert_allclose(metrics["total_return"],
                                   metrics["buyhold_return"], atol=1e-6)
