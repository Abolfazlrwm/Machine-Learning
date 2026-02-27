import pandas as pd
import numpy as np

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for stock price prediction.
    Indicators: Returns, SMA, EMA, RSI, MACD, Volatility.
    """
    df = df.copy()
    
    # 1. Daily Return
    df['Return'] = df['Close'].pct_change()
    
    # 2. Simple Moving Averages (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
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
    
    # 6. Volatility (Standard Deviation of returns over 10 days)
    df['Volatility'] = df['Return'].rolling(window=10).std()
    
    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the binary target variable:
    1 if next day's Close > today's Close, else 0.
    """
    # Shift close prices by -1 to get "tomorrow's" price on "today's" row
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Target: 1 if Up, 0 if Down
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Drop rows with NaN values created by indicators and shifting
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