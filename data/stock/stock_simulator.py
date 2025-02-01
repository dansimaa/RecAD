from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np
from data.stock.configs.stock_config import SimulationConfig
from data.stock.corr_matrix import CorrelationMatrix
from data.utils.random_utils import RandomUtils


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

    def __init__(self, config: SimulationConfig, random_seed: int):
        self.config = config
        self.random_seed = random_seed
        self.n_days = self.config.n_years * self.config.trading_days_per_year

        correlation_matrix = CorrelationMatrix(
            self.config.corr_config_path
        ).get_corr_matrix()
        
        self.simulation_model = BlackScholesMonteCarlo(
            correlation_matrix, 
            self.random_seed
        )

    def _generate_random_parameters(
            self
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate initial stock prices, drift, and volatility values."""
        RandomUtils.set_seed(self.random_seed)
        S0 = self.config.s0 + np.random.randn(self.config.n_stocks)
        mu = np.random.uniform(
            self.config.mu[0], self.config.mu[1], self.config.n_stocks
        )
        sigma = np.random.uniform(
            self.config.sigma[0], self.config.sigma[1], self.config.n_stocks
        )
        return S0, mu, sigma
    
    def simulate_stock_prices(self) -> np.ndarray:
        """Simulate stock prices using the chosen model."""
        S0, mu, sigma = self._generate_random_parameters()
        stock_prices = self.simulation_model.run(
            S0, mu, sigma, self.config.n_years, self.n_days
        )
        return stock_prices 
