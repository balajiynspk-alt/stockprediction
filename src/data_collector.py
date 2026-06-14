import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.logger import setup_logger

logger = setup_logger("data_collector")

class StockDataCollector:
    """
    Class responsible for collecting historical stock market data using the yfinance library.
    """
    def __init__(self, cache_dir: str = "data"):
        """
        Initializes the StockDataCollector with a cache directory.
        """
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

    def fetch_data(self, ticker: str, years: int = 5, force_download: bool = False) -> pd.DataFrame:
        """
        Fetches historical stock data for the given ticker.
        If cache exists and force_download is False, it loads from cache.
        Otherwise, downloads from Yahoo Finance.
        
        Parameters:
        -----------
        ticker : str
            The stock ticker symbol (e.g. 'AAPL', 'RELIANCE.NS').
        years : int
            Number of years of historical data to fetch. Default is 5.
        force_download : bool
            If True, ignores cached files and downloads fresh data.
            
        Returns:
        --------
        pd.DataFrame
            Historical stock data containing Date index and standard columns:
            Open, High, Low, Close, Adj Close, Volume.
        """
        ticker = ticker.upper().strip()
        cache_file = os.path.join(self.cache_dir, f"{ticker}_historical.csv")
        
        # Check cache first
        if not force_download and os.path.exists(cache_file):
            logger.info(f"Loading data for {ticker} from cache: {cache_file}")
            try:
                # Read CSV and ensure Date is set as index
                df = pd.read_csv(cache_file, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                
                # Check if cache is recent enough (e.g., within 1 day) and has at least the required years of data
                if not df.empty:
                    min_date = df.index.min()
                    max_date = df.index.max()
                    target_start_date = datetime.now() - timedelta(days=years * 365)
                    
                    if min_date <= target_start_date and (datetime.now() - max_date).days <= 1:
                        logger.info(f"Cached data is valid and covers the requested time range ({min_date.date()} to {max_date.date()}).")
                        return df
                    else:
                        logger.info("Cached data is either outdated or doesn't cover the full 5-year period. Re-downloading...")
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file} due to: {e}. Downloading new data.")
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Candidate list to try: [original, original.NS, original.BO]
        candidates = [ticker]
        if "." not in ticker:
            candidates.append(f"{ticker}.NS")
            candidates.append(f"{ticker}.BO")
            
        df = pd.DataFrame()
        resolved_ticker = None
        
        for candidate in candidates:
            logger.info(f"Downloading historical data for candidate ticker '{candidate}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            try:
                stock = yf.Ticker(candidate)
                df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                
                if df.empty:
                    df = yf.download(candidate, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
                
                if not df.empty:
                    resolved_ticker = candidate
                    break
            except Exception as ex:
                logger.warning(f"Attempt failed for candidate '{candidate}': {ex}")
                continue
                
        if df.empty:
            raise ValueError(f"No data could be retrieved for ticker '{ticker}' (tried fallbacks: {candidates}). The ticker might be invalid or there is no network connection.")
            
        logger.info(f"Successfully resolved and retrieved data using ticker: '{resolved_ticker}'")
        
        try:
            # Clean column names (sometimes columns are MultiIndex or have spaces)
            if isinstance(df.columns, pd.MultiIndex):
                # Clean multi-index columns which yfinance sometimes returns for download
                df.columns = df.columns.get_level_values(0)
            
            # Reset index and make sure 'Date' is standard datetime, then set index back
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            # Make index name consistent
            df.index.name = 'Date'
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                # If Adj Close is present and Close is missing, rename it
                if 'Adj Close' in df.columns and 'Close' not in df.columns:
                    df.rename(columns={'Adj Close': 'Close'}, inplace=True)
                    logger.info("Renamed 'Adj Close' to 'Close'")
                    missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise KeyError(f"Downloaded data is missing required columns: {missing_cols}")
            
            # Remove Timezone information to avoid conflicts with datetime operations
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            # Save to cache
            df.to_csv(cache_file)
            logger.info(f"Successfully downloaded and saved data for {ticker} (shape: {df.shape}) to {cache_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error post-processing data for resolved ticker '{resolved_ticker}': {e}")
            raise e
