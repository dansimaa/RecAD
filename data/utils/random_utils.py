import numpy as np


class RandomUtils:
    """Utility class for handling random operations."""
    @staticmethod
    def set_seed(seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)