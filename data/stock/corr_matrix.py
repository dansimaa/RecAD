import warnings
from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from data.stock.configs.corr_matrix_config import CorrelationConfig


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
                raise KeyError(f"Price field '{price_field}' \
                               not found for stock '{stock}'.")
        return df


class CorrelationMatrixCalculator:
    """Calculates the correlation matrix of the given stock data."""

    def __init__(self, stock_data: pd.DataFrame):
        self.stock_data = stock_data

    def compute_corr_matrix(self, method: str = "pearson") -> np.array:
        """Compute the correlation matrix of the log returns."""
        daily_returns = self.stock_data.pct_change().dropna()
        log_returns = np.log(1 + daily_returns)
        corr_matrix = log_returns.corr(method=method)
        return corr_matrix.to_numpy()
     

class CorrelationMatrix:
    """Manages the end-to-end process of computing stock correlation matrix."""
        
    def __init__(self, config_path: Path):
        self.config = CorrelationConfig.from_yaml(config_path)
        self.config.validate()
        self.stock_data = self._fetch_stock_data()

        self._corr_matrix = self._compute_corr_matrix()
        self._check_corr_matrix() 
    
    def _fetch_stock_data(self) -> pd.DataFrame:
        """Retrieve stock data using the StockDataFetcher."""
        fetcher = StockDataFetcher(
            stocks=self.config.stocks,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        return fetcher.download_data(self.config.price_field)
    
    def _compute_corr_matrix(self) -> np.ndarray:
        """Compute and retrieve the correlation matrix."""
        calculator = CorrelationMatrixCalculator(self.stock_data)
        return calculator.compute_corr_matrix()

    @staticmethod
    def _is_valid_correlation_matrix(matrix: np.ndarray) -> bool:
        """Validate the correlation matrix."""
        return (
            isinstance(matrix, np.ndarray) and
            matrix.shape[0] == matrix.shape[1] and
            np.allclose(matrix, matrix.T) and
            np.allclose(np.diagonal(matrix), 1) and
            not np.isnan(matrix).any()
        )
    
    def _check_corr_matrix(self) -> None:
        if not self._is_valid_correlation_matrix(self._corr_matrix):
            warnings.warn(
                "The computed correlation matrix is invalid.", UserWarning
            )
    
    def get_corr_matrix(self) -> np.ndarray:
        """Getter method for the correlation matrix."""
        return self._corr_matrix
    