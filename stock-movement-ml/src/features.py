import pandas as pd
import numpy as np

# Columns used as model input. Kept in one place so features.py, train.py
# and backtest.py never drift apart.
FEATURE_COLUMNS = [
    "Return", "Return_lag1", "Return_lag2", "Return_lag3",
    "SMA_10", "SMA_20", "Dist_SMA20",
    "EMA_20", "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Width",
    "ATR_14", "OBV_Ratio", "Vol_Ratio", "Range_Pct", "Volatility",
]


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for stock price prediction.

    Indicators: Returns (+lags), SMA/EMA, distance to SMA, RSI, MACD,
    Bollinger Bands, ATR, OBV ratio, volume ratio, daily range, volatility.

    Adds columns without mutating the input frame.
    """
    df = df.copy()

    # 1. Daily Return + lagged returns (momentum features)
    df['Return'] = df['Close'].pct_change()
    for lag in (1, 2, 3):
        df[f'Return_lag{lag}'] = df['Return'].shift(lag)

    # 2. Simple Moving Averages (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Dist_SMA20'] = df['Close'] / df['SMA_20'] - 1.0  # % distance to SMA20

    # 3. Exponential Moving Average (EMA)
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # 4. Relative Strength Index (RSI) - 14 days
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. Moving Average Convergence Divergence (MACD)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 6. Bollinger Bands (20-day, 2 std)
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + 2.0 * std_20
    df['BB_Lower'] = df['SMA_20'] - 2.0 * std_20
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']

    # 7. Average True Range (14 days) - volatility of the daily range
    prev_close = df['Close'].shift(1)
    true_range = pd.concat(
        [
            df['High'] - df['Low'],
            (df['High'] - prev_close).abs(),
            (df['Low'] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean() / df['Close']  # relative ATR

    # 8. On-Balance Volume (normalised by a 20-day rolling sum)
    direction = np.sign(df['Close'].diff()).fillna(0.0)
    obv = (direction * df['Volume']).cumsum()
    obv_sum = (direction * df['Volume']).rolling(window=20).sum().abs()
    df['OBV_Ratio'] = obv.diff() / obv_sum.replace(0, np.nan)

    # 9. Volume ratio vs its 20-day average
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 10. Daily range (High-Low) relative to Close
    df['Range_Pct'] = (df['High'] - df['Low']) / df['Close']

    # 11. Volatility (Std of returns over 10 days)
    df['Volatility'] = df['Return'].rolling(window=10).std()

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the binary target variable:
    1 if next day's Close > today's Close, else 0.

    Also drops rows with NaNs produced by rolling indicators and the
    final row (which has no "next day" yet) -> no look-ahead leakage.
    """
    df = df.copy()
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    df = df.dropna()
    return df


if __name__ == "__main__":
    # Example usage (for testing purposes)
    try:
        raw_data = pd.read_csv("data/AAPL_raw_data.csv", index_col=0)
        df_features = calculate_technical_indicators(raw_data)
        df_final = create_target(df_features)
        print(f"Features created. Final shape: {df_final.shape}")
        print(df_final[['Close', 'Target']].head())
    except FileNotFoundError:
        print("Raw data file not found. Run data_loader.py first.")
