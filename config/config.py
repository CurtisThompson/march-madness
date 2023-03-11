import yaml

def get_config(config_path='./config/config.yaml'):
    """Imports a config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config