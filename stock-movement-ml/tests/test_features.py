import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, calculate_technical_indicators, create_target


class TestCalculateTechnicalIndicators:
    def test_all_feature_columns_present(self, ohlcv):
        df = calculate_technical_indicators(ohlcv)
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        assert not missing, f"missing columns: {missing}"

    def test_input_not_mutated(self, ohlcv):
        before = ohlcv.copy()
        calculate_technical_indicators(ohlcv)
        pd.testing.assert_frame_equal(ohlcv, before)

    def test_rsi_bounds(self, ohlcv):
        rsi = calculate_technical_indicators(ohlcv)["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_definition(self, ohlcv):
        df = calculate_technical_indicators(ohlcv)
        recomputed = df["MACD"] - df["MACD_Signal"]
        np.testing.assert_allclose(df["MACD_Hist"], recomputed, rtol=1e-9)

    def test_sma_matches_manual(self, ohlcv):
        df = calculate_technical_indicators(ohlcv)
        manual = ohlcv["Close"].rolling(10).mean()
        np.testing.assert_allclose(df["SMA_10"], manual, rtol=1e-9)

    def test_return_lags(self, ohlcv):
        df = calculate_technical_indicators(ohlcv)
        np.testing.assert_allclose(df["Return_lag2"], df["Return"].shift(2),
                                   atol=0, rtol=1e-12)


class TestCreateTarget:
    def test_target_direction(self):
        df = pd.DataFrame(
            {"Close": [10.0, 11.0, 9.0],
             "High": [10.5, 11.5, 9.5],
             "Low": [9.5, 10.5, 8.5],
             "Open": [10.0, 10.5, 10.0],
             "Volume": [100, 100, 100]},
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        result = create_target(df)
        # day0 -> day1 up => 1 ; day1 -> day2 down => 0 ; last row dropped
        assert result["Target"].tolist() == [1, 0]
        assert len(result) == 2  # last row has no "next day"

    def test_no_nan_after_dropna(self, features_df):
        assert not features_df.isna().any().any()

    def test_no_lookahead_in_last_row(self, ohlcv):
        df = create_target(calculate_technical_indicators(ohlcv))
        assert df.index[-1] == ohlcv.index[-2], \
            "final row must be dropped (its target would need tomorrow's close)"
