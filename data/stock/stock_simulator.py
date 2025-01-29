from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
from data.stock.corr_matrix import CorrelationMatrix
from data.utils.random_utils import RandomUtils

TRADING_DAYS_PER_YEAR: int = 250


class SimulationModel(ABC):
    """Abstract base class for simulation models."""
    
    @abstractmethod
    def run(self, 
            S0: np.ndarray, 
            mu: np.ndarray, 
            sigma: np.ndarray, 
            n_years: int, 
            n_days: int) -> np.ndarray:
        pass


class BlackScholesMonteCarlo(SimulationModel):
    """Black-Scholes Monte Carlo model for correlated stock price simulation."""

    def __init__(self, corr_matrix: np.ndarray, seed: int):
        self.corr_matrix = corr_matrix
        self.seed = seed

    def run(self, 
            S0: np.ndarray, 
            mu: np.ndarray, 
            sigma: np.ndarray, 
            n_years: int, 
            n_days: int) -> np.ndarray:
        """Run MC simulations of stock prices using the Black-Scholes model."""
        RandomUtils.set_seed(self.seed)
        N = len(S0)
        dt = n_years / n_days

        # Compute Cholesky decomposition for correlated Brownian motion
        C = np.linalg.cholesky(self.corr_matrix)
        X = np.zeros((N, n_days + 1))

        for i in range(n_days):
            Z = np.random.randn(N)
            Y = np.matmul(C, Z)
            X[:, i + 1] = X[:, i] + (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Y

        return S0[:, None] * np.exp(X[:, 1:])


class StockSimulator:
    """Class responsible for simulating stock prices based on configurations."""

    def __init__(self, config: Dict):
        self.config = config
        self.random_seed = config["random_seed"]
        self.n_stocks = config["simulation"]["n_stocks"]
        self.n_years = config["simulation"]["n_years"]
        self.n_days = self.n_years * TRADING_DAYS_PER_YEAR 
        self.s0 = config["simulation"]["s0"]
        self.mu_range = config["simulation"]["mu"]
        self.sigma_range = config["simulation"]["sigma"]

        correlation_matrix = CorrelationMatrix(
            config["simulation"]["corr_config_path"]
        ).get_corr_matrix()
        
        self.simulation_model = BlackScholesMonteCarlo(
            correlation_matrix, 
            self.random_seed
        )

    def _generate_random_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial stock prices, drift, and volatility values."""
        RandomUtils.set_seed(self.random_seed)
        S0 = self.s0 + np.random.randn(self.n_stocks)
        mu = self.mu_range[0] + (self.mu_range[1] - self.mu_range[0]) * np.random.rand(self.n_stocks)
        sigma = self.sigma_range[0] + (self.sigma_range[1] - self.sigma_range[0]) * np.random.rand(self.n_stocks)
        return S0, mu, sigma
    
    def simulate_stock_prices(self) -> np.ndarray:
        """Simulate stock prices using the chosen model."""
        S0, mu, sigma = self._generate_random_parameters()
        stock_prices = self.simulation_model.run(S0, mu, sigma, self.n_years, self.n_days)
        return stock_prices 
