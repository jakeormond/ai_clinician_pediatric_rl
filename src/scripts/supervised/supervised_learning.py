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
from ...utils.logging_utilities import wandb_init  # Add this import

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train mortality prediction model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    parser.add_argument('--model_type', 
                        default='xgb', 
                        choices=['xgb'], 
                        help='Model type to train (default: xgb)')
    parser.add_argument('--fold', 
                        type=int, 
                        default=0, 
                        help='Fold number to train (default: 0)')
    parser.add_argument('--mode', type=str, default='single_fold',
                        choices=['single_fold', 'aggregated'],
                        help='Mode of training: single_fold or aggregated (default: single_fold)')
    parser.add_argument("--base_experiment_dir", type=str,
                       help="Base experiment directory (required for aggregate mode)")
    
    return parser.parse_args()


def validate_arguments(args):
    if args.mode == "single_fold" and args.fold is None:
        raise ValueError("--fold is required when mode is 'single_fold'")
    if args.mode == "aggregate" and args.base_experiment_dir is None:
        raise ValueError("--base_experiment_dir is required when mode is 'aggregate'")


 
def main():

    args = parse_arguments()
    validate_arguments(args)

    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(
        hospital=args.hospital, 
        experiment_type="supervised_learning",
        model_type=args.model_type)

    if args.mode == "single_fold":

        experiment_name = f"mortality_{args.hospital}_{args.model_type}_fold{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}"
    
        # update config with fold and experiment name
        config_manager.add_fold(args.fold)
        experiment_dir = config_manager.add_experiment_name(experiment_name)

        # load the search grid and cross validation settings

        # Initialize W&B with experiment-specific config
        wandb_config = config_manager.get_wandb_config(
            experiment_type="supervised_learning",
            hospital=args.hospital,
            model_type=args.model_type,
            seed=config["seed"],
            fold=args.fold,
            group=f"{args.hospital}_supervised",
            tags=["mortality", args.hospital, args.model_type],
            config={
                "n_outer_folds": config["cv_config"]["n_outer_folds"],  # From config file
                "n_inner_folds": config["cv_config"]["n_inner_folds"],  # From config file
            }
        )
        
        wandb_init(wandb_config)

        dataset = Dataset(config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")
            
        # train the model using the outer training loop
        experiment_name = f"mortality_{args.model_type}_grid_search_{time.strftime('%Y%m%d_%H%M%S')}_fold{args.fold}"
        experiment_dir = config_manager.create_experiment_dir(experiment_name)

        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {experiment_dir}")

        outer_training_loop(dataset, config, experiment_dir, args.model_type, target_fold = args.fold)
    

if __name__ == "__main__":
    main()