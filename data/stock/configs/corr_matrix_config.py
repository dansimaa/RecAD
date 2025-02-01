from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import yaml


@dataclass
class CorrelationConfig:
    """Configuration for correlation matrix computation."""
    stocks: List[str]
    start_date: str
    end_date: str
    price_field: str = "Adj Close"

    @classmethod
    def from_dict(cls, config: Dict) -> 'CorrelationConfig':
        """Creates a CorrelationConfig instance from a dictionary."""
        return cls(
            stocks=config["stocks"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            price_field=config["price_field"]
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'CorrelationConfig':
        """Loads configuration from a YAML file."""
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def validate(self):
        """Validates the configuration values."""
        if not self.stocks or not isinstance(self.stocks, list):
            raise ValueError("Stocks must be a non-empty list.")
        if not isinstance(self.start_date, str) or not isinstance(self.end_date, str):
            raise ValueError("Start and end dates must be strings.")
