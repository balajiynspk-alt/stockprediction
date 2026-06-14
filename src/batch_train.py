import os
import time
import argparse
import pandas as pd
from src.logger import setup_logger
from src.train import run_pipeline

logger = setup_logger("batch_train")

def run_batch_training(limit: int = None, tune: bool = False, years: int = 5):
    csv_path = "data/nse_equities.csv"
    if not os.path.exists(csv_path):
        logger.error(f"NSE equities list not found at '{csv_path}'. Please run the Streamlit app or download it first.")
        return
        
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        tickers = df['SYMBOL'].str.strip().dropna().unique().tolist()
    except Exception as e:
        logger.error(f"Failed to read NSE equities file: {e}")
        return

    logger.info(f"Loaded {len(tickers)} symbols from '{csv_path}'.")
    
    if limit is not None and limit > 0:
        logger.info(f"Limiting batch training to the first {limit} tickers.")
        tickers = tickers[:limit]
        
    results = []
    
    # Create directory for models and images if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Processing ticker: {ticker}")
        start_time = time.time()
        
        try:
            # Append .NS directly to optimize query speed and bypass fallbacks
            resolved_symbol = f"{ticker}.NS"
            
            # Run training pipeline
            model, engineer, metrics, df_forecast = run_pipeline(
                ticker=resolved_symbol,
                years=years,
                tune=tune
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Successfully trained model for {resolved_symbol} in {elapsed:.2f} seconds.")
            
            results.append({
                "Ticker": ticker,
                "Resolved_Symbol": resolved_symbol,
                "Status": "Success",
                "R2_Score": metrics.get("R2", None),
                "RMSE": metrics.get("RMSE", None),
                "MAE": metrics.get("MAE", None),
                "MAPE": metrics.get("MAPE", None),
                "Training_Time_Sec": elapsed,
                "Error": ""
            })
            success_count += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed to train model for {ticker}: {e}")
            results.append({
                "Ticker": ticker,
                "Resolved_Symbol": f"{ticker}.NS",
                "Status": "Failed",
                "R2_Score": None,
                "RMSE": None,
                "MAE": None,
                "MAPE": None,
                "Training_Time_Sec": elapsed,
                "Error": str(e)
            })
            fail_count += 1
            
        # Tiny delay to avoid rate-limiting from Yahoo Finance API
        time.sleep(1.0)
        
    # Save training summary
    summary_df = pd.DataFrame(results)
    summary_path = "data/batch_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    logger.info("=" * 60)
    logger.info("BATCH TRAINING COMPLETE SUMMARY:")
    logger.info(f"Total Tickers Attempted: {len(tickers)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch train models for NSE equities")
    parser.add_argument("--limit", type=str, default="5", help="Number of tickers to train (e.g. 5, 50, or 'all')")
    parser.add_argument("--tune", action="store_true", help="Enable GridSearch hyperparameter tuning (slower)")
    parser.add_argument("--years", type=int, default=5, help="Years of historical data to fetch")
    
    args = parser.parse_args()
    
    limit_val = None
    if args.limit.lower() != "all":
        try:
            limit_val = int(args.limit)
        except ValueError:
            logger.warning(f"Invalid limit value '{args.limit}'. Training all tickers.")
            
    run_batch_training(limit=limit_val, tune=args.tune, years=args.years)
