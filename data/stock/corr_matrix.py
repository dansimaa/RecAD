from typing import List, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from data.utils.loaders import ConfigLoader


class StockDataFetcher:
    """Handles downloading stock data."""
    def __init__(self, stocks: List[str], start_date: str, end_date: str):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self, price_field: str = "Adj Close") -> pd.DataFrame:
        """Download stock data for the given tickers and return a DataFrame of prices."""
        df = pd.DataFrame()
        for stock in tqdm(self.stocks, desc="Downloading stock data"):
            stock_data = yf.download(
                stock, start=self.start_date, end=self.end_date, progress=False
            )
            if price_field in stock_data:
                df[stock] = stock_data[price_field]
            else:
                raise KeyError(f"Price field '{price_field}' not found for stock '{stock}'.")
        return df


class CorrelationMatrixCalculator:
    """Calculates the correlation matrix of the given stock data."""

    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data

    def compute_corr_matrix(self) -> np.array:
        """Compute the correlation matrix of the log returns."""
        daily_returns = self.stock_data.pct_change().dropna()
        log_returns = np.log(1 + daily_returns)
        corr_matrix = log_returns.corr()
        return corr_matrix.to_numpy()
     

class CorrelationMatrix:
    """Manages the end-to-end process of computing stock correlation matrix."""
        
    def __init__(self, config_path: str):
        self.config = self._load_and_validate_config(config_path)
        self.stock_data = self._fetch_stock_data()

        self._corr_matrix = self._compute_corr_matrix()

    @staticmethod
    def _load_and_validate_config(config_path: str) -> Dict:
        """Load and validate configuration."""
        config = ConfigLoader.load_config(config_path)
        ConfigLoader.validate_config(config, ["start_date", "end_date", "stocks"])
        return config    
    
    def _fetch_stock_data(self) -> pd.DataFrame:
        """Retrieve stock data using the StockDataFetcher."""
        fetcher = StockDataFetcher(
            stocks=self.config["stocks"],
            start_date=self.config["start_date"],
            end_date=self.config["end_date"],
        )
        return fetcher.download_data()
    
    def _compute_corr_matrix(self) -> np.ndarray:
        """Compute and retrieve the correlation matrix."""
        calculator = CorrelationMatrixCalculator(self.stock_data)
        return calculator.compute_corr_matrix()

    def get_corr_matrix(self) -> np.ndarray:
        """Getter method for the correlation matrix."""
        return self._corr_matrix
