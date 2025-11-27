import os
import yaml
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import random
import itertools
import joblib
import glob
import copy
import numpy as np
import pandas as pd
import pickle
import shap
import wandb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, GroupKFold, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from scipy.special import softmax
from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score
)
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from .evaluation import calculate_metrics
# from .plotting import create_all_plots, create_aggregated_shap_plots
from ..utils.logging_utilities import wandb_init
from ..data.data_processor import DataProcessor
from ..data.data_processor_for_shards import ShardProcessor
# from .dataset import Dataset
from ..rl.replay_buffer import DynamicDeepHitDataset, DynamicDeepHit_Lazy, worker_init_fn

from ..utils.utilities import set_seed
from ..rl.plotting import plot_losses,plot_losses_cv, plot_c_and_brier_over_time, plot_auroc_prc_curves
from .dynamic_deephit_evaluation import compute_survival_metrics

import logging
logger = logging.getLogger(__name__)


# --------------------- Serialization Utilities ---------------------
def to_json_safe(obj):
    """Recursively convert numpy arrays, torch tensors, and numpy scalars
    into JSON-serializable Python types (lists, floats, ints).
    """
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        # detach and move to cpu before converting
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    # numpy scalar types
    if hasattr(obj, 'item') and (
        isinstance(obj, (np.generic,))
    ):
        return obj.item()
    return obj

MODEL_MODULES = {
    "dynamic_deephit": "src.supervised_learning.models.dynamic_deephit",
}

DISTRIBUTIONS = {
    "loguniform": loguniform,
    "uniform": uniform,
}

def parse_distribution(dist_str):
    # Example: "loguniform(1e-5, 1e-2)"
    name, args = dist_str.split("(", 1)
    args = args.rstrip(")")
    args = [float(x) for x in args.split(",")]
    return DISTRIBUTIONS[name](*args)


def average_dicts(list_of_dicts):
    """Averages values for each key across a list of dictionaries.
    Using this for evaluation metrics.    
    """

    from collections import defaultdict

    # Collect all values for each key
    values = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            values[k].append(v)

    # Compute averages
    averages = {k: sum(v_list) / len(v_list) for k, v_list in values.items()}
    return averages

def get_fixed_hyperparameters(config_manager):
    """Extracts fixed hyperparameters from config_manager."""
    hyperparameters = config_manager.training_config.copy()
    return hyperparameters

def select_hyperparameters_randomly(hyperparameter_space, config_manager):
    """Selects hyperparameters randomly from the given space."""
    selected_params = {}
    for param, value in hyperparameter_space.items():
        if isinstance(value, list):
            selected_params[param] = random.choice(value)
        # else if it is a single value
        elif isinstance(value, (int, float, str, bool)):
            selected_params[param] = value
        else:  # assume it's a scipy.stats distribution
            selected_params[param] = round(value.rvs(), 6)  # round for readability
    return selected_params


def create_processors(dataset, train_id, val_id, config_manager):
    """Creates training and validation DataProcessors with shared normalization."""
    
    train_dataset = dataset.create_subset(dataset.encounter_ids()[train_id])     
    val_dataset = dataset.create_subset(dataset.encounter_ids()[val_id])
    
    train_processor = DataProcessor(train_dataset, config_manager)
    norm_params = train_processor.get_normalization_params()
    val_processor = DataProcessor(val_dataset, config_manager, 
                                 norm_params=norm_params)
    
    return train_processor, val_processor, norm_params


def create_processor_from_shards(config_manager, normalize=True):
    train_processor = ShardProcessor(config_manager, normalize=normalize, split='train')
    val_processor = ShardProcessor(config_manager, split='val',
                                  norm_params=train_processor.get_normalization_params())
    test_processor = ShardProcessor(config_manager, split='test',
                                   norm_params=train_processor.get_normalization_params())

    return train_processor, val_processor, test_processor

def create_buffer(processor, batch_size, config_manager):
    if config_manager.hospital == 'choc':
        dataset = DynamicDeepHitDataset(processor.dynamic_deephit_dataset)
        buffer = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif config_manager.hospital == 'mimic_iv':
        dataset = DynamicDeepHit_Lazy(
            processor,
            config_manager
        )
        buffer = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4, # set to 0 for debugging; increase for performance (e.g., 4 or 8)
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )
    
    return buffer


def create_trainer(train_processor, hyperparams, config_manager):

    # feature_dim = train_processor.dynamic_deephit_dataset["features"].shape[2]
    feature_dim = train_processor.feature_dim
    num_risks = 2 # hardcode for now

    model_type = config_manager.model_type
    module = importlib.import_module(MODEL_MODULES[model_type])
    
    return module.get_trainer(
        config_manager,
        feature_dim,
        hyperparams,
        train_processor
    )


def get_early_stopping_config(config_manager):
    """Extracts early stopping configuration from config_manager."""
    early_stopping = {
        "patience": config_manager.training_config.get('patience', 10),
        "patience_counter": 0,
        "eval_freq": config_manager.training_config.get('eval_freq', 5),
        "min_delta": config_manager.training_config.get('min_delta', 0.001)
    }
    return early_stopping


def append_epoch_losses(
        epoch_training_losses, 
        training_losses
    ):
    """Calculates average losses for the epoch and appends to training_losses."""      
    epoch_avg_losses = {key: np.mean(values) for key, values in epoch_training_losses.items()}
    # Append epoch averages to training_losses
    for key, value in epoch_avg_losses.items():
        training_losses[key].append(value)     


def evaluate_model(
    trainer, 
    dataloader, 
    max_samples=None,
    return_full_curves=False,
    compute_survival_metrics_flag=True
):
    """
    Evaluate model on a dataset.
    
    Args:
        trainer: Model trainer instance
        dataloader: DataLoader for evaluation data
        config_manager: Configuration manager
        max_samples: Maximum number of samples to evaluate (None for all)
        compute_survival_metrics_flag: Whether to compute survival metrics
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    eval_losses = defaultdict(list)
    all_logits = []
    all_time_to_event = []
    all_event_labels = []
    
    num_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Check if we've reached max samples
        if max_samples is not None and num_samples >= max_samples:
            break
        
        batch_losses, logits = trainer.evaluate(batch)
        
        # Collect batch losses
        for key, value in batch_losses.items():
            eval_losses[key].append(value)
        
        # Collect data for survival metrics
        if logits is not None and compute_survival_metrics_flag:
            all_logits.append(logits.cpu())
            # all_time_to_event.append(batch[1].cpu())
            all_time_to_event.append(batch['time_to_event'].cpu())
            # all_event_labels.append(batch[2].cpu())
            all_event_labels.append(batch['event'].cpu())
        
        # num_samples += batch[0].size(0)
        num_samples += batch['features'].size(0)
    
    # Calculate average losses
    avg_losses = {key: np.mean(values) for key, values in eval_losses.items()}
    
    # Compute survival metrics if requested and data is available
    if all_logits and compute_survival_metrics_flag:
        logits = torch.cat(all_logits, dim=0)
        time_to_event = torch.cat(all_time_to_event, dim=0)
        event_labels = torch.cat(all_event_labels, dim=0)
        
        survival_metrics = compute_survival_metrics(
            logits,
            time_to_event,
            event_labels,
            return_full_curves=return_full_curves,
            no_censoring=True
        )
        
        avg_losses.update(survival_metrics)
    
    return avg_losses


def evaluate_during_training(trainer, epoch, val_buffer, val_losses, config_manager):
    """Evaluate model during training and log results."""
    
    # Evaluate with sample limit for speed
    max_samples = config_manager.training_config.get('max_val_samples', 1000)
    epoch_val_metrics = evaluate_model(
        trainer, 
        val_buffer, 
        max_samples=max_samples,
        return_full_curves=False,
        compute_survival_metrics_flag=True
    )
    
    # Log metrics
    if len(epoch_val_metrics) > 1:
        logger.info(f"Epoch {epoch}: Validation metrics: {epoch_val_metrics}")
    
    # Add epoch number
    epoch_val_metrics['epoch'] = epoch
    
    # Append to validation losses
    for key, value in epoch_val_metrics.items():
        # 

        if type(value) is torch.Tensor:
            if value.shape[0] > 1:
                value = value.detach().cpu().numpy()

        val_losses[key].append(value)


def evaluate_final_model(trainer, test_buffer):
    """Evaluate final model on test set."""
    
    # Evaluate on full test set (no sample limit)
    test_metrics = evaluate_model(
        trainer,
        test_buffer,
        max_samples=None,
        compute_survival_metrics_flag=True, 
        return_full_curves=True
    )
    
    logger.info(f"Final test metrics: {test_metrics}")
    
    return test_metrics


def check_improvement(best_val_loss, current_val_loss, early_stopping):
    improvement = best_val_loss - current_val_loss
    if improvement > early_stopping['min_delta']:
        best_val_loss = current_val_loss
        early_stopping['patience_counter'] = 0               
        logger.info(f"NEW BEST: Val loss={current_val_loss:.4f}")
    else:
        early_stopping['patience_counter'] += 1                
        logger.info(f"No improvement, Val={current_val_loss:.4f}, ({early_stopping['patience_counter']}/{early_stopping['patience']} - NO IMPROVEMENT)")
    
    return improvement, best_val_loss


def copy_model_parameters(trainer, config_manager):
    return trainer.model.state_dict().copy()
    

def restore_model_parameters(trainer, best_model_state, config_manager):
    trainer.model.load_state_dict(best_model_state) 
    

def single_split_training(config_manager):
    """
    Runs training on a single predefined train/val/test split, with data saved
    to HDF5 files (i.e. shards).
    This one doesn't use a Dataset class, just DataProcessors created from the HDF5 files. 
    On the first call, it normalizes the data and resaves the normalized data to new HDF5 files.
    """
    logger.info("Starting single split training...")

    # Create processors
    train_processor, val_processor, test_processor = create_processor_from_shards(
        config_manager, 
        normalize=True
        )

    hyperparameters = get_fixed_hyperparameters(config_manager)
    training_scores, test_scores, n_epochs, trainer = train_with_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager, 
            hyperparameters,
            n_epochs=config_manager.n_epochs,
            final_model=False
        )



    

    dataset = DynamicDeepHit_Lazy(train_processor, config_manager)   

    loader = DataLoader(
        dataset,
        batch_size=config_manager.training_config['batch_size'],
        shuffle=True,
        num_workers=4, # set to 0 for debugging; increase for performance
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    batch = next(iter(loader))


    pass



def development_training_loop(dataset, config_manager):
    """Runs a simplified training loop with 80-20 train/test split for development purposes."""
    
    logger.info("Starting development training loop...")

    # Create train/val/test split (80/10/10)
    n_samples = len(dataset.encounter_ids())
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_end = int(0.8 * n_samples)
    
    train_id = indices[:train_end]
    test_id = indices[train_end:]

    # Create processors
    train_processor, test_processor, norm_params = create_processors(
        dataset, train_id, test_id, config_manager
    )

    # train 
    hyperparameters = get_fixed_hyperparameters(config_manager)
    training_scores, test_scores, n_epochs, trainer = train_with_hyperparameters(
            train_processor, 
            test_processor, 
            config_manager, 
            hyperparameters,
            n_epochs=config_manager.n_epochs,
            final_model=False
        )

    # plot losses
    plot_losses(training_scores, test_scores, None, config_manager)

    if config_manager.model_type == 'dynamic_deephit':
        # plot the validation C-index and Brier scores over time
        plot_c_and_brier_over_time(test_scores, config_manager)

        # evaluate final model and plot auroc and average precision curves
        test_buffer = create_buffer(test_processor, config_manager.training_config['batch_size'], config_manager)
        final_test_scores = evaluate_final_model(
            trainer,
            test_buffer
            )
        plot_auroc_prc_curves(
            final_test_scores['roc_curve'], 
            final_test_scores['pr_curve'],
            final_test_scores['prevalence'],
            config_manager
            )
        

    logger.info("Development training loop complete.")
    logger.info(f"Test scores: {test_scores}")

    # save the configuration_manager
    with open(config_manager.experiment_dir / "config_manager.pkl", "wb") as f:
        pickle.dump(config_manager, f)  

    # create a dictionary to save the scores
    all_scores = {
        "training_scores": training_scores,
        "test_scores": test_scores,
        "hyperparameters": hyperparameters,
        "n_epochs": n_epochs
    }

    # save model so that it can be reloaded later
    model_path = Path(config_manager.experiment_dir /  "final_model.pt")
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f" Final model saved to {model_path}")

    with open(config_manager.experiment_dir / "all_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)

    # save the normalization parameters as a pkl file
    with open(config_manager.experiment_dir / "norm_params.pkl", "wb") as f:
        pickle.dump(norm_params, f)        
        
    # Save results for this outer fold
    results_path = Path(config_manager.experiment_dir) / f"results.json"
    with open(results_path, 'w') as f:
        json.dump(to_json_safe({
            'hyperparameters': hyperparameters,
            'final_metrics': test_scores,
            'config': config_manager.training_config
        }), f, indent=2)
        
        logger.info(f"Training complete. Results saved to {results_path}")
          
    return {
        "training_scores": training_scores,
        "test_scores": test_scores,
        }


def plot_results(training_losses, val_losses, test_losses, config_manager):

    plot_losses(training_losses, val_losses, test_losses, config_manager)

    if config_manager.model_type == 'dynamic_deephit':
        # plot the validation C-index and Brier scores over time
        plot_c_and_brier_over_time(val_losses, config_manager)

        plot_auroc_prc_curves(
            test_losses['roc_curve'], 
            test_losses['pr_curve'],
            test_losses['prevalence'],
            config_manager
            )
        

def save_results(
        training_scores, 
        val_scores, 
        final_training_losses, 
        val_losses, 
        test_losses, 
        best_hyperparameters, 
        n_epochs, 
        norm_params, 
        config_manager
    ):
    
    outer_fold = config_manager.cv_config.get('fold', 'N/A')

    # save the configuration_manager
    with open(config_manager.experiment_dir / "config_manager.pkl", "wb") as f:
        pickle.dump(config_manager, f)  

    # create a dictionary to save the scores
    all_scores = {
        "inner_training_scores": training_scores,
        "inner_val_scores": val_scores,
        "final_training_scores": final_training_losses,
        "final_test_scores_across_training": val_losses, 
        "final_test_scores": test_losses,
        "best_hyperparameters": best_hyperparameters,
        "n_epochs": n_epochs
    }
    with open(config_manager.experiment_dir / "all_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)

    # save the normalization parameters as a pkl file
    with open(config_manager.experiment_dir / "norm_params.pkl", "wb") as f:
        pickle.dump(norm_params, f)        
        
    # Save results for this outer fold
    results_path = Path(config_manager.experiment_dir) / f"fold_{outer_fold}_results.json"
    # drop the curves from test_losses for saving
    if 'roc_curve' in test_losses:
        test_losses.pop('roc_curve')
    if 'pr_curve' in test_losses:
        test_losses.pop('pr_curve')

    with open(results_path, 'w') as f:
        json.dump(to_json_safe({
            'outer_fold': outer_fold,
            'best_hyperparameters': best_hyperparameters,
            'final_metrics': test_losses,
            'config': config_manager.training_config
        }), f, indent=2)
    
    logger.info(f"✓ Outer fold {outer_fold} complete. Results saved to {results_path}")

    return all_scores


def outer_training_loop(dataset, config_manager): 
    """Runs single outer fold - designed for HPC job array."""
    
    logger.info("Starting outer training loop...")

    # Get CV configuration
    n_outer_folds, n_inner_folds, target_fold = config_manager.load_cv_config()
    
    logger.info(f"Processing outer fold {target_fold+1}/{n_outer_folds}")
    
    # Outer CV split
    kf_outer = KFold(n_splits=n_outer_folds, shuffle=True, 
                     random_state=config_manager.data_config["seed"])

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if outer_fold != target_fold:
            continue  # Skip other folds
                   
        # Create outer train/test splits
        train_dataset = dataset.create_subset(dataset.encounter_ids()[outer_train_id])     
        test_dataset = dataset.create_subset(dataset.encounter_ids()[outer_test_id])

        ###################   HYPERPARAMETER OPTIMIZATION #############################
        logger.info("Phase 1: Hyperparameter Optimization")
        best_hyperparameters, training_scores, val_scores, n_epochs = optimize_hyperparameters(
            train_dataset, 
            config_manager, 
            n_inner_folds=n_inner_folds
        )
        logger.info(f"Best hyperparameters: {best_hyperparameters}")

        # get the median number of epochs used in inner folds
        n_epochs = int(np.median(n_epochs)) # for final training

        #################### RETRAIN FINAL MODEL #############################
        logger.info("Phase 2: Final Model Training & Evaluation")
        
        # create train and test processors with shared normalization
        train_processor, test_processor, norm_params = create_processors(
                dataset, outer_train_id, outer_test_id, config_manager
            )        
                        
        final_training_losses, \
        final_val_losses, \
        test_losses = train_with_hyperparameters(
            train_processor, 
            test_processor, 
            config_manager, 
            best_hyperparameters,
            n_epochs=n_epochs,
            final_model=True
        )

        # PLOT THE TRAINING AND VALIDATION LOSSES
        plot_results(final_training_losses, final_val_losses, test_losses, config_manager)

        ##################### SAVE RESULTS #############################
        all_scores = save_results(
            training_scores,
            val_scores,
            final_training_losses,
            final_val_losses,
            test_losses,
            best_hyperparameters,
            n_epochs,
            norm_params,
            config_manager
        )
                
        return all_scores
    

def get_hyperparameter_space(config_manager):
    """Creates hyperparameter space from config_manager."""
    hyperparameter_space = {}

    # loop through config_manager.training_config
    for k, v in config_manager.training_config.items():
        if isinstance(v, str) and '(' in v and ')' in v:
            hyperparameter_space[k] = parse_distribution(v)
        else: # everything else
            hyperparameter_space[k] = v
    
    # check if any values are lists or distributions, throw error if not
    if not any(isinstance(v, list) or (isinstance(v, str) and '(' in v and ')' in v) 
               for v in config_manager.training_config.values()):
        raise ValueError("No hyperparameters to optimize. Ensure that config_manager.training_config contains lists of values to search over or distribution strings.")

    return hyperparameter_space


def optimize_hyperparameters(
    train_dataset, 
    config_manager,
    n_inner_folds=None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[int]]:
    """Inner CV loop for hyperparameter optimization."""
    
    loss_metric = config_manager.training_config['loss_metric'] # e.g. actor loss, critic loss etc, depends on algorithm

    ################################ create hyperparameter space ################################
    # the hyperparameter space to search consists of the parameters that are lists 
    # in config_manager.training_config
    hyperparameter_space = get_hyperparameter_space(config_manager)

    ############################ initialize best_val_score #############################
    # used to select best hyperparameters
    best_val_loss = float('inf') 
    
    ##################### loop through hyperparameter combinations #####################
    logger.info(f"Starting hyperparameter optimization with {config_manager.n_search_iterations} iterations...")
    
    for i in range(config_manager.n_search_iterations):
        ################# select hyperparameters randomly #################
        current_params = select_hyperparameters_randomly(
            hyperparameter_space, config_manager
        )       
        logger.info(f"Testing hyperparameters {i+1}/{config_manager.n_search_iterations}: {current_params}")
        
        # create storage for validation, training scores, and n_epochs
        n_inner_folds_to_run = config_manager.cv_config.get('n_inner_to_run', n_inner_folds)
        inner_training_scores = [None] * n_inner_folds_to_run
        inner_val_scores = [None] * n_inner_folds_to_run
        inner_n_epochs = [None] * n_inner_folds_to_run        

        ################# Inner CV evaluation for testing hyperparameters #################       
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True, 
                         random_state=config_manager.data_config["seed"])        
        
        for inner_fold, (inner_train_id, inner_val_id) in enumerate(kf_inner.split(train_dataset.encounter_ids())):
            if inner_fold >= config_manager.cv_config.get('n_inner_to_run', n_inner_folds):
                logger.info(f"  Skipping inner fold {inner_fold+1} as per n_inner_to_run setting.")
                continue  # Skip extra inner folds if specified
            
            logger.info(f"  Inner fold {inner_fold+1}/{n_inner_folds}")
            
            # create train and val processors with shared normalization
            inner_train_processor, inner_val_processor, _ = create_processors(
                train_dataset, inner_train_id, inner_val_id, config_manager
            )
           
            ######################### Train with current hyperparameters ########################
            inner_training_scores[inner_fold], \
            inner_val_scores[inner_fold], \
            inner_n_epochs[inner_fold],\
                _ = train_with_hyperparameters(
                inner_train_processor, 
                inner_val_processor, 
                config_manager, 
                current_params
            )                        
            
            current_val_loss = min(inner_val_scores[inner_fold][loss_metric])
            logger.info(f"  Inner fold {inner_fold+1} val loss: {current_val_loss:.4f}")
        
        ###################### Evaluate and update best hyperparameters ########################
        overall_val_loss = np.mean([min(score[loss_metric]) for score in inner_val_scores])
        logger.info(f" Overall mean validation {loss_metric}: {overall_val_loss:.4f}")
        if overall_val_loss < best_val_loss:
            improvement = best_val_loss - overall_val_loss
            logger.info(f" New best hyperparameters found! (improvement: {improvement:.4f})\n")
            best_val_loss = overall_val_loss
            best_hyperparams = current_params.copy()
            training_scores_best = inner_training_scores
            val_scores_best = inner_val_scores
            n_epochs_best = inner_n_epochs
        else:
            logger.info(f"No improvement (current best: {best_val_loss:.4f})\n")
        
    ############################ Summarize hyperparameter optimization results #############################
    logger.info(f"Best hyperparameters after {config_manager.n_search_iterations} iterations: {best_hyperparams}")
    logger.info(f"\n                 Hyperparameter optimization complete!")

    return best_hyperparams, training_scores_best, val_scores_best, n_epochs_best


def get_hyperparameter(hyperparams, config_manager, param_name):
    """Get hyperparameter value from hyperparams or config_manager."""
    if param_name in hyperparams:
        return hyperparams[param_name]
    else:
        return config_manager.training_config[param_name]


def train_with_hyperparameters(
        train_processor, 
        val_test_processor, 
        config_manager, 
        hyperparams, 
        n_epochs = None, 
        final_model=False
    ):
    """Train model with specific hyperparameters and return validation score."""
    
    if n_epochs is None:
        n_epochs = config_manager.n_epochs

    batch_size = get_hyperparameter(hyperparams, config_manager, 'batch_size')
    loss_metric = config_manager.training_config['loss_metric']

    # Create replay buffers
    training_buffer = create_buffer(train_processor, batch_size, config_manager)
    val_test_buffer = create_buffer(val_test_processor, batch_size, config_manager)
    
    # create trainer
    trainer = create_trainer(train_processor, hyperparams, config_manager)
        
    # Early stopping configuration 
    early_stopping = get_early_stopping_config(config_manager)

    best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training with early stopping (patience={early_stopping['patience']}, eval_freq={early_stopping['eval_freq']}, min_delta={early_stopping['min_delta']})")
    
    # Use defaultdict to automatically create new keys
    training_losses = defaultdict(list) 
    
    # Evaluate before training starts to get initial validation loss
    val_losses = defaultdict(list)
    # evaluate before any training for epoch 0
    evaluate_during_training(trainer, -1, val_test_buffer, val_losses, config_manager)   
    
    ############################### Training loop #################################
    for epoch in range(n_epochs):
        epoch_training_losses = defaultdict(list)

        for batch_idx, batch in enumerate(training_buffer):
            batch_losses, _ = trainer.train(batch)

            # Automatically append all metrics - creates new keys if needed
            for key, value in batch_losses.items():
                epoch_training_losses[key].append(value)

        # Calculate average loss for this epoch and append to training_losses
        append_epoch_losses(epoch_training_losses, training_losses)

        #################### Evaluate ####################
        # note: this evaluation is used for early stopping during hyperparameter tuning,
        # but only used for visualization during final model training
        # Evaluate periodically
        if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                or epoch == n_epochs - 1:
            evaluate_during_training(trainer, epoch, val_test_buffer, val_losses, config_manager)
            
            # Determine validation loss for early stopping
            current_val_loss = val_losses[loss_metric][-1]            
            logger.info(f"Epoch {epoch}: Train {loss_metric}={training_losses[loss_metric][-1]:.4f}, Val {loss_metric}={current_val_loss:.4f}")
                            
            if not final_model: 
                # Check for improvement and copy model parameters if best
                improvement, best_val_loss = check_improvement(best_val_loss, current_val_loss, early_stopping)

                if improvement > 0:
                    best_model_state = copy_model_parameters(trainer, config_manager)
                    
                #  Early stopping - stop training THIS hyperparameter combination
                if early_stopping['patience_counter'] >= early_stopping['patience']:                
                    logger.info(f" Early stopping at epoch {epoch}/{n_epochs}")
                    logger.info(f" Best validation loss: {best_val_loss:.4f}")

                    n_epochs = epoch + 1  # Update n_epochs to actual number run

                    break
    
    ######################### return results if hyperparameter tuning ########################
    if not final_model:
        # Restore best model state before returning score
        if best_model_state is not None:
            restore_model_parameters(trainer, best_model_state, config_manager)
            logger.info(f"    Restored best model (val_loss={best_val_loss:.4f})")

        return training_losses, val_losses, n_epochs, trainer

    ########################## evaluate on test set after training if final model ##########################
    # note that results df will be None if no targets/predictions were collected during evaluation (i.e.Q learning)
    test_losses = evaluate_final_model( # this evaluates on the entire test set (during training we only use a subset for speed)
        trainer, 
        val_test_buffer
    )
    
    logger.info(f"Final test metrics: {test_losses}")

    # Save final model if this is the final training
    model_path = Path(config_manager.experiment_dir /  "final_model.pt")
    torch.save(trainer.model.state_dict(), model_path)
    logger.info(f" Final model saved to {model_path}")

    # Save test predictions if available
    # if results_df is not None:
    #     results_path = Path(config_manager.fold_dir / "test_predictions.parquet")
    #     results_df.to_parquet(results_path)
    #     logger.info(f" Test predictions saved to {results_path}\n")       

    return training_losses, val_losses, test_losses

