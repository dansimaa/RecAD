import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm


class StockDataFetcher:
    """Handles downloading stock data."""
    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        """Download stock data for the given tickers and return a DataFrame of adjusted close prices."""
        df = pd.DataFrame()
        for stock in tqdm(self.stocks, desc="Downloading stock data"):
            stock_data = yf.download(stock, 
                                     start=self.start_date, 
                                     end=self.end_date, 
                                     progress=False)
            df[stock] = stock_data["Adj Close"]
        return df


class CorrelationMatrix:
    """Calculates the correlation matrix of the given stock data."""
        
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self._validate_config()

        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.stocks = self.config["stocks"]

        # Fetch stock data using the StockDataFetcher class
        self.stock_data_fetcher = StockDataFetcher(self.stocks, self.start_date, self.end_date)
        self.stock_data = self._get_stock_data()
        self.corr_matrix = self._get_corr_matrix()

    def _load_config(self, config_path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _validate_config(self):
        """Ensure required configuration parameters are present."""
        required_keys = ["start_date", "end_date", "stocks"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")

    def _get_stock_data(self):
        """Retrieve stock data using the StockDataFetcher."""
        return self.stock_data_fetcher.download_data()

    def _get_corr_matrix(self):
        """Compute the correlation matrix of the log returns."""
        daily_returns = self.stock_data.pct_change().dropna()
        log_returns = np.log(1 + daily_returns)
        return log_returns.corr()

    def get_corr_matrix(self):
        """Public method to access the correlation matrix."""
        return self.corr_matrix.to_numpy()
