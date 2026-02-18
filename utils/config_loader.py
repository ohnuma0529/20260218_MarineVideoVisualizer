import yaml
import os

def load_config(config_path="config/default_config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at {config_path}. Using empty dict.")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            return {}

def merge_configs(base_config, override_args):
    """
    Merge command line arguments into the loaded config.
    """
    # Simple merge logic: override base_config with non-None values from args
    # This can be made more sophisticated as needed.
    pass
