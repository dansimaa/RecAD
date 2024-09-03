import yaml


class ConfigLoader:
    """Handles loading and processing of configuration files."""
    @staticmethod
    def load_config(config_path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)