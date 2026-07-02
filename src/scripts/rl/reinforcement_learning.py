import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import os
import time
import logging
import random


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

from ...rl.training import single_split_training
# from ...analyse_input_features.analyse_health_score import analyse_health_score

TensorBatch = List[torch.Tensor]

logger = logging.getLogger(__name__)

def setup_logger(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
#     # The following ensure complete reproducibility but can slow down training:
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False



def parse_arguments():
    parser = argparse.ArgumentParser(description='Train RL model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic_iv'], 
                        help='Hospital dataset to use (default: choc)')
    
    parser.add_argument('--experiment_type', 
                        default='reinforcement_learning', 
                        choices=['supervised_learning', 'reinforcement_learning'], 
                        help='Type of experiment to run (default: reinforcement_learning)')
    
    parser.add_argument('--model_type', 
                        default='any_percent_bc', 
                        choices=[
                            'any_percent_bc', 'td3_bc', 'basic_q', 'reward_analysis',
                            'conservative_q_learning', 'cql_with_logprobs', 'cvp', 'iql', 
                            'iql_with_sarsa'
                            ], 
                        help='Model type to train (default: any_percent_bc)') 

    parser.add_argument('--mode', type=str, default='single_split',
                        choices=['single_split', 'aggregate', 'analysis'],
                        help='Mode of training: single_split or aggregate (default: single_split)')
    

    parser.add_argument('--testing', action='store_true',                 
                help='if true, we pare down the data for quick testing')

        
    parser.add_argument('--split_num', type=int, nargs='+', default=None,
                        help='Split number to train (not strictly required for single_split mode)')    
    
    # parser.add_argument('--seed', type=int, default=[42],
    #                     help='Random seed to use for training (default: [42])')
    
    # mode-specific arguments    
    parser.add_argument('--fold', type=int, default=None,
                       help='Fold number to train (required for single_fold mode)')

    # parser.add_argument('--experiment_variant', 
    #                     default='quick_test', 
    #                     choices=['quick_test', 'standard'], 
    #                     help='Variant of the experiment to run (default: quick_test)')

    parser.add_argument('--search_variant', 
                        default='random_search', 
                        choices=['random_search', 'grid_search', 'fixed_hyperparameters'], 
                        help='Variant of the search to use (default: random_search)')

    parser.add_argument('--n_search_iterations', type=int, default=2,
                        help='Number of search iterations to run (default: 2)')
    
    parser.add_argument('--start_time', type=str, default = None,                  
                    help='Start time for this experiment run')
      
    parser.add_argument('--verbose', action='store_true',                 
                    help='if true, print details of all the features and normalization steps')
    
    parser.add_argument('--n_epochs', type=int, default = 10,                  
                    help='Number of epochs to train for - should be 1000, but for testing we set to 10')
    
    parser.add_argument('--cohort', type=str, default=None,
        choices=['all', 'sepsis_strict', 'sepsis_broad', 'sepsis',
            'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad'],
        help='Cohort to use for this experiment (overrides config file)')  
    
    parser.add_argument('--model_dir', type=str, default=None,
        help='Directory of the model to load for prediction or feature evaluation (required for prediction and feature_evaluation experiment variants)')
    
    parser.add_argument('--action_mode', 
                        type=str, 
                        default = "continuous_only",   
                        choices=['continuous_only', 'binary_only', 'mixed'],               
                        help='Type of action space to use - continuous_only, binary_only, or mixed')

    parser.add_argument('--frac', type=float, default = 0.,                  
                    help='Fraction of data to use for behaviour cloning - 0. means use all data')
    
    args = parser.parse_args()

    # if testing is true, make experiment_variant "quick_test", otherwise "standard"
    if args.testing:
        args.experiment_variant = "quick_test"
    else:
        args.experiment_variant = "standard"

    # If start_time is not provided, generate a timestamp
    if args.start_time is None:
        args.start_time = time.strftime("%Y%m%d_%H%M%S")

    if len(args.split_num) == 1: args.split_num = args.split_num[0]

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

    setup_logger(verbose=args.verbose)

    # # set random seed for reproducibility
    # set_seed(args.seed)

    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(args)

    if args.mode == "single_fold":
        # check that reward is included
        check_for_reward_config(config_manager)

        # create the name and directory for this fold
        config_manager.set_fold_name() # check code, we now now want a parent dir, with inner date dir

        # load data. Note that features, actions, and rewards can be used immediately for xgboost,
        # but require further processing for RL. 
        dataset = Dataset(config_manager.data_config)
        logger.info(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        if config_manager.testing:
            dataset = dataset.subsample_dataset(0.1)
            logger.info(f"Subsampled dataset to {len(dataset.dataframe)} rows for testing")            

        # Initialize W&B - will implement later
        # log_fold_results_to_wandb(results)

        # load training config
        config_manager.set_training_config() # based on model type 

        logger.info(f"Running on {config_manager.environment} environment")
        logger.info(f"Experiment directory: {config_manager.experiment_dir}")
        logger.info(f"Fold directory: {config_manager.fold_dir}\n")

        results = outer_training_loop(dataset, config_manager)

        # LOG TO W&B in main script
        # log_fold_results_to_wandb(results)

        logger.info(f"Fold {args.fold} completed successfully")
        logger.info(f"Test metrics: {results}")
        
        if wandb.run is not None:
            wandb.finish()
    

    elif args.mode == "single_split":
        config_manager.set_training_config()
        single_split_training(config_manager)



    elif args.mode == "analysis":
        dataset = Dataset(config_manager.data_config)
        logger.info(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        config_manager.set_training_config() # based on model type 

        # set config_manager.experiment_dir as "reward_analysis" within the experiments dir
        config_manager.experiment_dir = os.path.join(config_manager.paths['experiments_dir'], "reward_analysis")

        logger.info(f"Running on {config_manager.environment} environment")
        logger.info(f"Experiment directory: {config_manager.experiment_dir}")
        
        analyse_health_score(dataset, config_manager)

    
if __name__ == "__main__":
    main()