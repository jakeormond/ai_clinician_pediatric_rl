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
import wandb

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
    parser.add_argument('--mode', type=str, default='single_fold',
                        choices=['single_fold', 'aggregated'],
                        help='Mode of training: single_fold or aggregated (default: single_fold)')
    
    # mode-specific arguments    
    parser.add_argument('--fold', type=int,
                       help='Fold number to train (required for single_fold mode)')
    
    # Config file overrides (optional)
    parser.add_argument('--experiment_variant', type=str, default='standard',
                       choices=['standard', 'quick_test', 'full_search', 'production'],
                       help='Experiment variant determines config defaults')
    
    
    return parser.parse_args()


def validate_arguments(args):
    if args.mode == "single_fold" and args.fold is None:
        raise ValueError("--fold is required when mode is 'single_fold'")
    if args.mode == "aggregate" and args.base_experiment_dir is None:
        raise ValueError("--base_experiment_dir is required when mode is 'aggregate'")


def log_fold_results_to_wandb(results):
    """Log fold results to W&B."""
    if wandb.run is None:  # Check if W&B is active
        return
    
    fold = results['fold']
    
    # Use global wandb state
    wandb.log({
        f"fold_{fold}/best_cv_score": results['best_cv_score'],
        **{f"fold_{fold}/{k}": v for k, v in results['test_metrics'].items()}
    }, step=fold)


def main():

    args = parse_arguments()
    validate_arguments(args)

    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(
        hospital=args.hospital, 
        experiment_type="supervised_learning",
        model_type=args.model_type,
        experiment_variant=args.experiment_variant
        )

    if args.mode == "single_fold":        
        # set base experiment name and make directory, if it doesn't exist (it will 
        # already if another folder was run)
        config_manager.set_experiment_name()
        
        # update config with n_fodls and current fold 
        config_manager.set_cv_config() # sets the number of outer and inner folds
        config_manager.add_fold(args.fold)
        
        # create the name and directory for this fold
        config_manager.set_fold_name()
        
        # update config with the search space
        config_manager.set_search_grid()      
       
        # load data
        dataset = Dataset(config_manager.hospital_config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")
            
        # Initialize W&B 
        wandb_init(config_manager.wandb_config)

        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {config_manager.experiment_dir}")
        print(f"Fold directory: {config_manager.fold_dir}")

        results = outer_training_loop(dataset, config_manager)

        # LOG TO W&B in main script
        log_fold_results_to_wandb(results)

        print(f"Fold {args.fold} completed successfully")
        print(f"Test metrics: {results['test_metrics']}")
        
        if wandb.run is not None:
            wandb.finish()
    

if __name__ == "__main__":
    main()