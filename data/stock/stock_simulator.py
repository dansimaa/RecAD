import os
import numpy as np
import yaml
from abc import ABC, abstractmethod

from data.stock.corr_matrix import CorrelationMatrix
from data.utils.config_loader import ConfigLoader
from data.utils.random_utils import RandomUtils


class SimulationModel(ABC):
    """Abstract base class for simulation models."""
    
    @abstractmethod
    def run(self, S0, mu, sigma, n_years, n_days):
        pass

class BlackScholesMonteCarlo(SimulationModel):
    """Black-Scholes Monte Carlo model for stock price simulation."""

    def __init__(self, corr_matrix, seed):
        self.corr_matrix = corr_matrix
        self.seed = seed

    def run(self, S0, mu, sigma, n_years, n_days):
        """Run Monte Carlo simulations of stock prices using the Black-Scholes model."""
        RandomUtils.set_seed(self.seed)
        N = len(S0)
        dt = n_years / n_days
        C = np.linalg.cholesky(self.corr_matrix)
        X = np.zeros((N, n_days + 1))

        for i in range(n_days):
            Z = np.random.randn(N)
            Y = np.matmul(C, Z)
            X[:, i + 1] = X[:, i] + (mu - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Y

        return S0[:, None] * np.exp(X[:, 1:])

class StockSimulator:
    """Class responsible for simulating stock prices based on configurations."""

    def __init__(self, config):
        self.config = config
        self.seed = config["random_seed"]
        self.n_stocks = config["n_stocks"]
        self.n_years = config["n_years"]
        self.n_days = self.n_years * 250 
        self.s0 = config["s0"]
        self.mu_range = config["mu"]
        self.sigma_range = config["sigma"]
        self.corr_config_path = config["corr_config_path"]

        self.corr_matrix = CorrelationMatrix(self.corr_matrix_path).get_corr_matrix()
        self.simulation_model = BlackScholesMonteCarlo(self.corr_matrix, self.seed)

    def _generate_random_parameters(self):
        """Generate initial stock prices, drift, and volatility values."""
        RandomUtils.set_seed(self.seed)
        S0 = self.s0 + np.random.randn(self.n_stocks)
        mu = self.mu_range[0] + (self.mu_range[1] - self.mu_range[0]) * np.random.rand(self.n_stocks)
        sigma = self.sigma_range[0] + (self.sigma_range[1] - self.sigma_range[0]) * np.random.rand(self.n_stocks)
        return S0, mu, sigma
    
    def simulate_stock_prices(self):
        """Simulate stock prices using the chosen model."""
        S0, mu, sigma = self._generate_random_parameters()
        stock_prices = self.simulation_model.run(S0, mu, sigma, self.n_years, self.n_days)
        return stock_prices 
