import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def split_data(df: pd.DataFrame, train_size: float = 0.8):
    """
    Splits the data into training and testing sets based on time.
    No shuffling to preserve the temporal order of stock prices.
    """
    # Define features and target
    features = ['Return', 'SMA_10', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal', 'Volatility']
    X = df[features]
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
    
    # 1. Logistic Regression (Baseline)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic_Regression'] = lr
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random_Forest'] = rf
    
    # 3. XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    print("All models trained successfully!")
    return models

if __name__ == "__main__":
    # This block is for independent testing of this module
    try:
        # Assuming data with features already exists from Step 2
        data = pd.read_csv("data/AAPL_processed.csv") # Placeholder for demo
        X_train, X_test, y_train, y_test = split_data(data)
        trained_models = train_models(X_train, y_train)
    except FileNotFoundError:
        print("Processed data not found. Please run the full pipeline.")