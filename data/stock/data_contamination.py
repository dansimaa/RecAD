from typing import Tuple, Dict
import numpy as np
from data.utils.random_utils import RandomUtils
from data.stock.configs.stock_config import ContaminationConfig


class DataContaminator:
    """
    Class responsible for introducing anomalies into a given time series dataset.
    """

    def __init__(self, config: ContaminationConfig, random_seed: int) -> None:
        self.config = config
        self.random_seed = random_seed 

    def inject_anomalies(
            self, 
            time_series_data: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce anomalies into the time series data X.

        Parameters:
        - time_series_data: Time series data where each row is a separate time series.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Contaminated time series and corresponding anomaly labels.
        """
        RandomUtils.set_seed(self.random_seed)

        n_anomalies = self.config.n_anomalies
        n_time_series, n_time_steps = time_series_data.shape                  
        anomaly_mask = np.ones((n_time_series, n_time_steps))
        labels = np.zeros((n_time_series, n_time_steps))

        for i in range(n_time_series):
            u = np.random.rand(n_anomalies)
            sgn = np.where(u >= 0.5, 1, -1)
            delta = np.random.uniform(
                self.config.amplitude_range[0],
                self.config.amplitude_range[1],
                n_anomalies
            )
            anomaly_positions = np.random.randint(
                0, n_time_steps, n_anomalies
            )
            for k in range(n_anomalies):
                anomaly_mask[i, anomaly_positions[k]] = 1 + sgn[k] * delta[k]
                labels[i, anomaly_positions[k]] = 1

        contaminated_series = time_series_data * anomaly_mask
        return contaminated_series, labels
