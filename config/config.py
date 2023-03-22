import yaml

def get_config(config_path='./config/config.yaml'):
    """
    Import a config file.

    Args:
        config_path: Path to yaml file that contains configuration.
    
    Returns:
        Dictionary of configuration key-value pairs.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config