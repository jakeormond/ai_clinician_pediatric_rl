"""
Prepares data for supervised learning by creating a dataset class.
Will handle cross validation splits and data normalization.
Splits data into X and y (as opposed to the observations, actions,
and rewards in the RL setting).

Create a DataSet object that holds the full dataset, and then 
CVSplit objects that hold the encounter IDs for the splits
and methods to access the data in the DataSet object.
"""

import argparse
import pandas as pd
import numpy as np
import time

from ...config_manager import ConfigManager  # Import the config manager
from ...supervised_learning.dataset import Dataset
from ...supervised_learning.training import outer_training_loop

 
def main():
    parser = argparse.ArgumentParser(description='Train mortality prediction model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    parser.add_argument('--model_type', 
                        default='xgb', 
                        choices=['xgb'], 
                        help='Model type to train (default: xgb)')
    args = parser.parse_args()

    # Initialize config manager
    config_manager = ConfigManager()

    hospital = args.hospital
    config = config_manager.load_hospital_config(hospital)
    # config_path = f"config/{hospital}_supervised.yaml"
    # config = load_config(config_path)

    dataset = Dataset(config)
    print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")
          
    # train the model using the outer training loop
    model_type = args.model_type
    experiment_name = f"mortality_{model_type}_grid_search_{time.strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = config_manager.create_experiment_dir(experiment_name)

    print(f"Running on {config_manager.environment} environment")
    print(f"Experiment directory: {experiment_dir}")

    # experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    # os.makedirs(experiment_dir, exist_ok=True)
    outer_training_loop(dataset, config, experiment_dir, model_type="xgb")
    
    pass

if __name__ == "__main__":
    main()