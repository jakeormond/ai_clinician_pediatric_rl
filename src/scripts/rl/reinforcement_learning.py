import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import os
import time


import pyrallis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


from ...config_manager import ConfigManager  # Import the config manager
from ...supervised_learning.dataset import Dataset
from ...utils.logging_utilities import wandb_init  # Add this import

# from src.data.data_processor import DataProcessor
# from src.rl.replay_buffer import ReplayBuffer

from ...rl.training_v2 import outer_training_loop


TensorBatch = List[torch.Tensor]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RL model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    
    parser.add_argument('--experiment_type', 
                        default='reinforcement_learning', 
                        choices=['supervised_learning', 'reinforcement_learning'], 
                        help='Type of experiment to run (default: reinforcement_learning)')
    
    parser.add_argument('--model_type', 
                        default='any_percent_bc', 
                        choices=['any_percent_bc'], 
                        help='Model type to train (default: any_percent_bc)') 

    parser.add_argument('--mode', type=str, default='single_fold',
                        choices=['single_fold', 'aggregate'],
                        help='Mode of training: single_fold or aggregate (default: single_fold)')
    
    # mode-specific arguments    
    parser.add_argument('--fold', type=int,
                       help='Fold number to train (required for single_fold mode)')

    parser.add_argument('--experiment_variant', 
                        default='quick_test', 
                        choices=['quick_test', 'standard'], 
                        help='Variant of the experiment to run (default: quick_test)')

    parser.add_argument('--search_variant', 
                        default='random_search', 
                        choices=['random_search', 'grid_search'], 
                        help='Variant of the search to use (default: random_search)')

    parser.add_argument('--n_search_iterations', type=int, default=2,
                        help='Number of search iterations to run (default: 2)')
    
    parser.add_argument('--start_time', type=str, default = None,                  
                    help='Start time for this experiment run')
    
    parser.add_argument('--testing', type=bool, default = False,                  
                    help='if true, we pare down the data for quick testing')
    
    parser.add_argument('--n_epochs', type=int, default = 10,                  
                    help='Number of epochs to train for - should be 1000, but for testing we set to 10')
    
    parser.add_argument('--frac', type=float, default = 0.,                  
                    help='Fraction of data to use for behaviour cloning - 0. means use all data')
    
    args = parser.parse_args()

    # If start_time is not provided, generate a timestamp
    if args.start_time is None:
        args.start_time = time.strftime("%Y%m%d_%H%M%S")

    return args 


def validate_arguments(args):
    if args.mode == "single_fold" and args.fold is None:
        raise ValueError("--fold is required when mode is 'single_fold'")
    if args.mode == "aggregate" and args.experiment_folder is None:
        raise ValueError("--experiment_folder is required when mode is 'aggregate'")
    

def check_for_reward_config(config_manager):
    if (not config_manager.data_config['use_reward'] and
        not config_manager.data_config['use_terminal_reward']):
        
        raise ValueError("For RL experiments, an hourly reward or terminal reward (or both) must be specified in the dataset config file.") 

# we can use pyrallis to parse command line arguments and load configurations when
# we're further along in the project
def main():

    args = parse_arguments()
    validate_arguments(args)

    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(args)

    if args.mode == "single_fold":
        # check that reward is included
        check_for_reward_config(config_manager)

        # create the name and directory for this fold
        config_manager.set_fold_name() # check code, we now now want a parent dir, with inner date dir

        # load data
        dataset = Dataset(config_manager.data_config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        if config_manager.testing:
            dataset = dataset.subsample_dataset(0.1)
            print(f"Subsampled dataset to {len(dataset.dataframe)} rows for testing")            

        # Initialize W&B - will implement later
        # log_fold_results_to_wandb(results)

        # load training config
        config_manager.set_training_config() # based on model type (i.e. any_percent_bc.yaml in this case)   

        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {config_manager.experiment_dir}")
        print(f"Fold directory: {config_manager.fold_dir}")

        results = outer_training_loop(dataset, config_manager)

        # LOG TO W&B in main script
        # log_fold_results_to_wandb(results)

        print(f"Fold {args.fold} completed successfully")
        print(f"Test metrics: {results}")
        
        if wandb.run is not None:
            wandb.finish()

    
if __name__ == "__main__":
    main()