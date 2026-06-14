import os
import argparse
import pandas as pd
from src.logger import setup_logger
from src.data_collector import StockDataCollector
from src.feature_engineer import FeatureEngineer
from src.model_pipeline import StockPredictionModel
from src.visualizer import StockVisualizer

logger = setup_logger("train_pipeline")

def run_pipeline(ticker: str, years: int = 5, tune: bool = True, output_model_path: str = None) -> tuple:
    """
    Runs the complete training and evaluation pipeline for a given stock ticker.
    """
    logger.info("=" * 60)
    logger.info(f"Starting Stock Price Prediction Pipeline for ticker: {ticker}")
    logger.info("=" * 60)
    
    # 1. Initialize modules
    collector = StockDataCollector(cache_dir="data")
    engineer = FeatureEngineer()
    visualizer = StockVisualizer(output_dir="images")
    
    # Set model save path
    if output_model_path is None:
        output_model_path = os.path.join("models", f"{ticker.lower()}_rf_model.pkl")
        
    # 2. Data Collection
    try:
        df_raw = collector.fetch_data(ticker, years=years)
    except Exception as e:
        logger.error(f"Failed to collect data: {e}")
        raise e
        
    # 3. Data Preprocessing & Feature Engineering
    df_clean = engineer.preprocess_data(df_raw)
    df_features = engineer.add_technical_indicators(df_clean)
    df_dataset = engineer.create_target_variable(df_features)
    
    # 4. Train-Test Split
    X_train, y_train, X_test, y_test = engineer.split_data(df_dataset, train_ratio=0.8)
    
    # 5. Feature Scaling
    X_train_scaled, X_test_scaled = engineer.scale_features(X_train, X_test, fit=True)
    
    # 6. Model Training & Hyperparameter Tuning
    model = StockPredictionModel(random_state=42)
    if tune:
        # Define grid search space
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4]
        }
        model.tune_hyperparameters(X_train_scaled, y_train, param_grid=param_grid)
    else:
        model.train(X_train_scaled, y_train)
        
    # 7. Model Evaluation
    metrics = model.evaluate(X_test_scaled, y_test)
    
    # 8. Save Model and Scaler
    model.save_model(output_model_path, engineer)
    
    # 9. Visualization
    predictions = model.predict(X_test_scaled)
    visualizer.plot_actual_vs_predicted(y_test, predictions, ticker)
    
    df_importance = model.get_feature_importances(engineer.feature_cols)
    visualizer.plot_feature_importance(df_importance, ticker)
    
    visualizer.plot_residuals(y_test, predictions, ticker)
    visualizer.plot_correlation_heatmap(df_features[engineer.feature_cols], ticker)
    
    # 10. Future Forecast (7 trading days)
    df_forecast = model.predict_future_days(df_features, engineer, days=7)
    
    logger.info("=" * 60)
    logger.info(f"Pipeline finished successfully for {ticker}.")
    logger.info("=" * 60)
    
    return model, engineer, metrics, df_forecast

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stock Price Prediction Model using Random Forest Regressor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, TSLA, RELIANCE.NS)")
    parser.add_argument("--years", type=int, default=5, help="Number of historical years of data to download")
    parser.add_argument("--no_tune", action="store_true", help="Disable hyperparameter tuning (runs faster with default params)")
    
    args = parser.parse_args()
    
    run_pipeline(
        ticker=args.ticker,
        years=args.years,
        tune=not args.no_tune
    )
