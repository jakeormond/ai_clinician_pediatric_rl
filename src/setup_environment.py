import os
import socket
import yaml
from pathlib import Path

def detect_environment():
    """
    Detect if we're running on local machine or HPC.
    """
    hostname = socket.gethostname()
    username = os.getenv('USER', '')
    
    # HPC detection logic (adjust based on your HPC characteristics)
    if 'festus' in hostname.lower() or username.startswith('bt'):
        return 'hpc'
    else:
        return 'local'

def load_paths():
    """
    Load paths based on the detected environment.
    """
    config_path = Path(__file__).parent.parent / "config" / "paths.yaml"
    
    with open(config_path, 'r') as f:
        paths_config = yaml.safe_load(f)
    
    env = detect_environment()
    print(f"Detected environment: {env}")
    
    paths = paths_config[env]
    
    # Create directories if they don't exist
    for path_name, path_value in paths.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)
        print(f"Ensured {path_name} exists: {path_value}")
    
    return paths, env

if __name__ == "__main__":
    paths, env = load_paths()
    print(f"Running on {env} environment")
    print(f"Paths: {paths}")