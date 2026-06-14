import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import setup_logger

logger = setup_logger("feature_engineer")

class FeatureEngineer:
    """
    Class responsible for data preprocessing and feature engineering,
    including technical indicators, train-test splitting, and scaling.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = []

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by handling missing values and duplicates.
        """
        logger.info("Starting data preprocessing...")
        df_clean = df.copy()
        
        # Remove duplicates in index or columns
        original_len = len(df_clean)
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        df_clean.drop_duplicates(keep='first', inplace=True)
        new_len = len(df_clean)
        
        if original_len != new_len:
            logger.info(f"Removed {original_len - new_len} duplicate rows.")
            
        # Handle missing values in OHLCV columns first
        # Use forward fill then backward fill to preserve continuity
        cols_to_check = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_check:
            if col in df_clean.columns:
                null_count = df_clean[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Found {null_count} missing values in column '{col}'. Filling them.")
                    df_clean[col] = df_clean[col].ffill().bfill()
                    
        return df_clean

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds technical indicators as features to the DataFrame:
        - SMA 10, SMA 20, SMA 50
        - EMA 10, EMA 20
        - RSI (14)
        - MACD (12, 26, 9)
        - Bollinger Bands (20, 2)
        - Daily Returns
        - Volatility (20-day rolling std of returns)
        - Volume Change
        """
        logger.info("Engineering technical indicators...")
        df_feats = df.copy()
        
        # Simple Moving Averages (SMA)
        df_feats['SMA_10'] = df_feats['Close'].rolling(window=10).mean()
        df_feats['SMA_20'] = df_feats['Close'].rolling(window=20).mean()
        df_feats['SMA_50'] = df_feats['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages (EMA)
        df_feats['EMA_10'] = df_feats['Close'].ewm(span=10, adjust=False).mean()
        df_feats['EMA_20'] = df_feats['Close'].ewm(span=20, adjust=False).mean()
        
        # Relative Strength Index (RSI) - standard 14 days
        delta = df_feats['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        
        # Wilder's smoothing technique for average gain and loss
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10) # Prevent division by zero
        df_feats['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = df_feats['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_feats['Close'].ewm(span=26, adjust=False).mean()
        df_feats['MACD'] = ema_12 - ema_26
        df_feats['MACD_Signal'] = df_feats['MACD'].ewm(span=9, adjust=False).mean()
        df_feats['MACD_Hist'] = df_feats['MACD'] - df_feats['MACD_Signal']
        
        # Bollinger Bands (BB)
        df_feats['BB_Middle'] = df_feats['Close'].rolling(window=20).mean()
        bb_std = df_feats['Close'].rolling(window=20).std()
        df_feats['BB_Upper'] = df_feats['BB_Middle'] + (bb_std * 2)
        df_feats['BB_Lower'] = df_feats['BB_Middle'] - (bb_std * 2)
        
        # Daily Returns
        df_feats['Daily_Return'] = df_feats['Close'].pct_change()
        
        # Volatility (Rolling 20-day standard deviation of daily returns)
        df_feats['Volatility'] = df_feats['Daily_Return'].rolling(window=20).std()
        
        # Volume Change
        df_feats['Volume_Change'] = df_feats['Volume'].pct_change()
        
        # Define the set of features to be used in modeling
        self.feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Daily_Return', 'Volatility', 'Volume_Change'
        ]
        
        # Handle the missing values and potential infinities (e.g., division by zero in volume change or returns)
        original_shape = df_feats.shape
        df_feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feats.dropna(subset=self.feature_cols, inplace=True)
        logger.info(f"Technical indicators added. Dropped rows with NaNs or Infinities. Shape changed from {original_shape} to {df_feats.shape}")
        
        return df_feats

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates target variable 'Target_Close' representing the next day's closing price.
        Shifts the closing price column by -1.
        """
        logger.info("Creating target variable 'Target_Close' (shift -1)...")
        df_target = df.copy()
        
        # Target Close is tomorrow's closing price
        df_target['Target_Close'] = df_target['Close'].shift(-1)
        
        # Save the last row (which has NaN target) for future prediction
        self.last_row = df_target.iloc[[-1]].copy()
        
        # Drop the last row from training data since we don't know the future close price yet
        df_target.dropna(subset=['Target_Close'], inplace=True)
        logger.debug(f"Target variable created. Shape: {df_target.shape}")
        
        return df_target

    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        """
        Splits data chronologically into train and test sets to prevent time series data leakage.
        """
        logger.info(f"Splitting data chronologically with train_ratio={train_ratio}...")
        
        # Ensure data is sorted by date index
        df_sorted = df.sort_index()
        
        split_idx = int(len(df_sorted) * train_ratio)
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        # Features and Targets
        X_train = train_df[self.feature_cols]
        y_train = train_df['Target_Close']
        
        X_test = test_df[self.feature_cols]
        y_test = test_df['Target_Close']
        
        logger.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        return X_train, y_train, X_test, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Fits standard scaler on training data and scales both training and test data.
        Returns scaled data as pandas DataFrames to keep column names.
        """
        logger.info("Scaling features using StandardScaler...")
        if fit:
            scaled_train = self.scaler.fit_transform(X_train)
        else:
            scaled_train = self.scaler.transform(X_train)
            
        scaled_test = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(scaled_train, index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaled_test, index=X_test.index, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled

    def scale_single_row(self, X_row: pd.DataFrame) -> pd.DataFrame:
        """
        Scales a single row of features using the fitted scaler.
        """
        scaled = self.scaler.transform(X_row)
        return pd.DataFrame(scaled, index=X_row.index, columns=X_row.columns)
