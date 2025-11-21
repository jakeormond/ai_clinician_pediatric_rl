import os
import socket
import yaml
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def detect_environment():
    """
    Detect if we're running on local machine or HPC.
    """
    hostname = socket.gethostname().lower()
    fqdn = socket.getfqdn().lower()
    username = os.getenv('USER', '')
    
    # HPC detection logic (adjust based on your HPC characteristics)
    if 'festus' in hostname.lower() or username.startswith('bt'):
        logging.info("Running on Bayreuth HPC")
        return 'bayreuth_hpc'
    
    elif 'cx3.hpc.ic.ac.uk' in fqdn:
        logging.info("Running on Imperial HPC")
        return 'imperial_hpc'
    
    else:
        logging.info("Running on local machine")
        return 'local'


def load_paths():
    """
    Load paths based on the detected environment.
    """
    config_path = Path(__file__).parent.parent / "config" / "environment" / "paths.yaml"
    
    with open(config_path, 'r') as f:
        paths_config = yaml.safe_load(f)
    
    env = detect_environment()
    print(f"Detected environment: {env}")
    
    paths = paths_config[env]
    
    # Create directories if they don't exist
    for path_name, path_value in paths.items():
        path_obj = Path(path_value)

        if path_name == 'experiments_dir':
            # Create experiments_dir if it doesn't exist
            print(f"Checking {path_name}: {path_value}")
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"Ensured {path_name} exists: {path_value}")

        elif path_name in ['data_dir', 'code_dir']:
            # Check that data_dir and code_dir exist, raise error if not
            if not path_obj.exists():
                raise FileNotFoundError(
                    f"Required directory does not exist: {path_name} = {path_value}\n"
                    f"Please ensure this directory exists before running the script."
                )
            print(f"Verified {path_name} exists: {path_value}")
            
        else:
            # For any other paths, just print them
            print(f"Using {path_name}: {path_value}")
  
    return paths, env

if __name__ == "__main__":
    paths, env = load_paths()
    print(f"Running on {env} environment")
    print(f"Paths: {paths}")