import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import setup_logger
from src.feature_engineer import FeatureEngineer

logger = setup_logger("model_pipeline")

class StockPredictionModel:
    """
    Class responsible for training, tuning, evaluating, and saving
    the Random Forest Regression model, as well as making future forecasts.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.best_params = {}
        self.metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the Random Forest model on the training data.
        """
        logger.info("Training Random Forest Regressor (default parameters)...")
        self.model.fit(X_train, y_train)
        logger.info("Model training completed.")

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict = None):
        """
        Performs hyperparameter tuning using GridSearchCV with TimeSeriesSplit to avoid lookahead bias.
        """
        logger.info("Starting hyperparameter tuning using GridSearchCV and TimeSeriesSplit...")
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4]
            }
            
        # TimeSeriesSplit is crucial for time-series validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed. Best Parameters: {self.best_params}")
        logger.info(f"Best cross-validation negative MAE: {grid_search.best_score_:.4f}")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the model on test data using MAE, MSE, RMSE, R2, and MAPE.
        """
        logger.info("Evaluating model on test data...")
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero by filtering out zero values if they exist
        y_test_arr = np.array(y_test)
        non_zero_mask = y_test_arr != 0
        mape = np.mean(np.abs((y_test_arr[non_zero_mask] - predictions[non_zero_mask]) / y_test_arr[non_zero_mask])) * 100
        
        self.metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        logger.info(f"Evaluation Metrics:")
        for metric_name, val in self.metrics.items():
            logger.info(f"  - {metric_name}: {val:.4f}")
            
        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions for the given features.
        """
        return self.model.predict(X)

    def get_feature_importances(self, feature_cols: list) -> pd.DataFrame:
        """
        Retrieves feature importance rankings from the trained Random Forest model.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance_list = []
        for i in indices:
            feature_importance_list.append({
                'Feature': feature_cols[i],
                'Importance': importances[i]
            })
            
        df_importance = pd.DataFrame(feature_importance_list)
        return df_importance

    def predict_future_days(self, df_history: pd.DataFrame, feature_engineer: FeatureEngineer, days: int = 7) -> pd.DataFrame:
        """
        Predicts the closing price for the next N trading days recursively.
        At each step:
        1. Predicts the next closing price.
        2. Appends the prediction to the history.
        3. Estimates the corresponding Open, High, Low, Volume for the next day.
        4. Re-calculates all technical indicators.
        5. Drops temporary indicators and proceeds to the next day.
        """
        logger.info(f"Running recursive multi-step forecasting for the next {days} trading days...")
        
        # Copy the last ~100 rows of history to ensure we have enough data to calculate rolling metrics (SMA 50, etc.)
        hist_df = df_history.tail(100).copy()
        
        # We need to drop the Target_Close column if it exists to prevent issues
        if 'Target_Close' in hist_df.columns:
            hist_df.drop(columns=['Target_Close'], inplace=True)
            
        future_predictions = []
        last_date = hist_df.index[-1]
        
        # Determine the trading calendar frequency (assuming standard weekdays)
        current_date = last_date
        
        for i in range(days):
            # 1. Standard technical analysis calculations for the current history
            # Create a temporary copy to run the feature engineering on
            temp_df = hist_df.copy()
            
            # Recalculate indicators for the whole slice to update the last row's indicators
            temp_df_indicators = feature_engineer.add_technical_indicators(temp_df)
            
            if temp_df_indicators.empty:
                logger.error("Failed to generate features for future step. History might be too short.")
                break
                
            # Extract the features for the last row (which represents the most recent state)
            X_last = temp_df_indicators[feature_engineer.feature_cols].tail(1)
            
            # Scale features
            X_last_scaled = feature_engineer.scale_single_row(X_last)
            
            # Predict the NEXT day's closing price
            pred_close = self.model.predict(X_last_scaled)[0]
            
            # Advance date to next business day
            next_date = current_date + pd.tseries.offsets.BDay(1)
            
            # Estimate Open, High, Low, Volume for this next day
            prev_close = hist_df.iloc[-1]['Close']
            pred_open = prev_close  # Flat opening assumption
            pred_high = max(pred_open, pred_close) * 1.002 # Add small margin
            pred_low = min(pred_open, pred_close) * 0.998  # Add small margin
            
            # Average volume of the last 5 days
            avg_volume = hist_df['Volume'].tail(5).mean()
            
            # Create the new row
            new_row = pd.DataFrame({
                'Open': [pred_open],
                'High': [pred_high],
                'Low': [pred_low],
                'Close': [pred_close],
                'Volume': [avg_volume]
            }, index=[next_date])
            
            new_row.index.name = 'Date'
            
            # Append new row to our rolling history
            hist_df = pd.concat([hist_df, new_row])
            
            # Save predictions
            future_predictions.append({
                'Day': i + 1,
                'Date': next_date.strftime('%Y-%m-%d'),
                'Predicted_Close': pred_close
            })
            
            # Move index forward
            current_date = next_date
            
        df_forecast = pd.DataFrame(future_predictions)
        logger.info(f"Future forecast completed: \n{df_forecast.to_string(index=False)}")
        return df_forecast

    def save_model(self, filepath: str, feature_engineer: FeatureEngineer):
        """
        Saves the trained model, scaler, feature columns, and evaluation metrics using pickle.
        """
        logger.info(f"Saving model artifacts to {filepath}...")
        
        # Ensure directories exist
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        artifacts = {
            'model': self.model,
            'scaler': feature_engineer.scaler,
            'feature_cols': feature_engineer.feature_cols,
            'metrics': self.metrics,
            'best_params': self.best_params
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(artifacts, f)
            logger.info("Model artifacts saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise e

    @classmethod
    def load_model(cls, filepath: str) -> tuple:
        """
        Loads the model and scaler from the specified file.
        Returns a tuple of (StockPredictionModel, FeatureEngineer).
        """
        logger.info(f"Loading model artifacts from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
            
        try:
            with open(filepath, 'rb') as f:
                artifacts = pickle.load(f)
                
            # Create instances
            model_instance = cls()
            model_instance.model = artifacts['model']
            model_instance.metrics = artifacts.get('metrics', {})
            model_instance.best_params = artifacts.get('best_params', {})
            
            feature_engineer = FeatureEngineer()
            feature_engineer.scaler = artifacts['scaler']
            feature_engineer.feature_cols = artifacts['feature_cols']
            
            logger.info("Model artifacts loaded successfully.")
            return model_instance, feature_engineer
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
