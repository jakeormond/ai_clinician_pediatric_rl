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
# from .dataset import Dataset
from ..rl.replay_buffer import DynamicDeepHitDataset

from ..utils.utilities import set_seed
from ..rl.plotting import plot_losses

import logging
logger = logging.getLogger(__name__)


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


def select_hyperparameters_randomly(hyperparameter_space, config_manager):
    """Selects hyperparameters randomly from the given space."""
    selected_params = {}
    for param, values in hyperparameter_space.items():
        if isinstance(values, list):
            selected_params[param] = random.choice(values)
        else:  # assume it's a scipy.stats distribution
            selected_params[param] = round(values.rvs(), 6)  # round for readability
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


def create_buffer(processor, batch_size):
    dataset = DynamicDeepHitDataset(processor.dynamic_deephit_dataset)
    buffer = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return buffer


def create_trainer(train_processor, hyperparams, config_manager):

    feature_dim = train_processor.dynamic_deephit_dataset["features"].shape[2]
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



def compute_survival_metrics(
    logits: np.ndarray,
    time_to_event: np.ndarray,
    event_label: np.ndarray,
    time_bins: np.ndarray
) -> Dict[str, float]:
    """
    Compute survival analysis metrics.
    
    Args:
        predictions: [batch_size, num_risks, num_time_bins] - predicted probabilities
        time_to_event: [batch_size] - actual time to event
        event_label: [batch_size] - event type (0=censored, 1=event1, 2=event2, etc.)
        time_bins: array of time bin edges
        
    Returns:
        Dictionary of survival metrics
    """
    metrics = {}

    # Convert logits to probabilities using softmax over time dimension
    # Shape: [batch_size, num_risks, num_time_bins]
    probs = softmax(logits, axis=-1)  # softmax over time bins    
    
    # Convert to structured array format required by scikit-survival
    # event_indicator: True if event occurred, False if censored
    event_indicator = event_label > 0
    
    # Create structured array
    y_true = np.array(
        [(bool(e), float(t)) for e, t in zip(event_indicator, time_to_event)],
        dtype=[('event', bool), ('time', float)]
    )
    
    # For C-index, we need a risk score (higher = higher risk)
    # Use predicted probability of event by the actual event time
    # Sum over all risks and time bins up to observed time
    risk_scores = []
    for i, t in enumerate(time_to_event):
        # Find the time bin index
        bin_idx = np.searchsorted(time_bins, t, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, probs.shape[2] - 1)

        risk_type = event_label[i]

        if risk_type == 0:
            # Censored - use overall event probability (sum across all risks)
            cif = probs[i, :, :bin_idx+1].sum() # check, this may be wrong!!!! if prob greater than 1!!!!
        else:
            # Event occurred - use CIF for the specific risk that occurred
            # CIF for risk k at time t = sum of probabilities for risk k up to time t
            cif = probs[i, risk_type-1, :bin_idx+1].sum()  # -1 because risk labels are 1-indexed
        
        
        # Sum probabilities across risks up to this time
        cif = probs[i, :, :bin_idx+1].sum()
        risk_scores.append(cif)
    
    risk_scores = np.array(risk_scores)
    
    # Compute C-index
    try:
        c_index = concordance_index_censored(
            event_indicator,
            time_to_event,
            risk_scores
        )
        metrics['c_index'] = c_index[0]  # Returns (c_index, concordant, discordant, tied)
    except Exception as e:
        logger.warning(f"Could not compute C-index: {e}")
        metrics['c_index'] = np.nan
    
    # Compute survival probabilities
    # S(t) = 1 - CIF(t) = 1 - sum over all risks of cumulative probabilities
    cumulative_incidence = np.cumsum(probs.sum(axis=1), axis=1)  # [batch, time_bins]
    survival_probs = 1 - cumulative_incidence

    # Compute Brier score at specific time points
    # Use the middle time bins as evaluation points
    eval_times = time_bins[::max(1, len(time_bins)//5)]  # Sample ~5 time points
    
    try:
        # Compute Brier score at each evaluation time
        brier_scores = []
        for eval_time in eval_times:
            if eval_time > time_to_event.max() or eval_time <= time_to_event.min():
                continue
                
            bin_idx = np.searchsorted(time_bins, eval_time, side='right') - 1
            if bin_idx >= survival_probs.shape[1] or bin_idx < 0:
                continue
                
            surv_prob_at_t = survival_probs[:, bin_idx]
            
            # Compute Brier score for this time point
            bs = brier_score(
                y_true,
                y_true,  # training data for IPCW estimation
                surv_prob_at_t,
                eval_time
            )
            brier_scores.append(bs[1])  # Returns (times, scores)
        
        if brier_scores:
            metrics['brier_score_mean'] = np.mean(brier_scores)
            metrics['brier_score_std'] = np.std(brier_scores)
        else:
            metrics['brier_score_mean'] = np.nan
            metrics['brier_score_std'] = np.nan
        
        # Integrated Brier Score over all time bins
        ibs = integrated_brier_score(
            y_true,
            y_true,  # training data for IPCW estimation
            survival_probs,
            time_bins[:survival_probs.shape[1]]
        )
        metrics['integrated_brier_score'] = ibs
        
    except Exception as e:
        logger.warning(f"Could not compute Brier scores: {e}")
        metrics['brier_score_mean'] = np.nan
        metrics['brier_score_std'] = np.nan
        metrics['integrated_brier_score'] = np.nan
    
    return metrics


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
    config_manager,
    max_samples=None,
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
            all_logits.append(logits.cpu().numpy())
            all_time_to_event.append(batch[1].cpu().numpy())
            all_event_labels.append(batch[2].cpu().numpy())
        
        num_samples += batch[0].size(0)
    
    # Calculate average losses
    avg_losses = {key: np.mean(values) for key, values in eval_losses.items()}
    
    # Compute survival metrics if requested and data is available
    if all_logits and compute_survival_metrics_flag:
        logits = np.vstack(all_logits)
        time_to_event = np.concatenate(all_time_to_event)
        event_labels = np.concatenate(all_event_labels)
        
        survival_metrics = compute_survival_metrics(
            logits,
            time_to_event,
            event_labels,
            config_manager.time_bins
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
        config_manager,
        max_samples=max_samples,
        compute_survival_metrics_flag=True
    )
    
    # Log metrics
    if len(epoch_val_metrics) > 1:
        logger.info(f"Epoch {epoch}: Validation metrics: {epoch_val_metrics}")
    
    # Add epoch number
    epoch_val_metrics['epoch'] = epoch
    
    # Append to validation losses
    for key, value in epoch_val_metrics.items():
        val_losses[key].append(value)


def evaluate_final_model(trainer, test_buffer, config_manager):
    """Evaluate final model on test set."""
    
    # Evaluate on full test set (no sample limit)
    test_metrics = evaluate_model(
        trainer,
        test_buffer,
        config_manager,
        max_samples=None,
        compute_survival_metrics_flag=True
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
    if config_manager.model_type in ['basic_q']:
        return trainer.critic.state_dict().copy()
    else:
        return trainer.actor.state_dict().copy()
    
def restore_model_parameters(trainer, best_model_state, config_manager):
    if config_manager.model_type in ['basic_q']:
        trainer.critic.load_state_dict(best_model_state)
    else:
        trainer.actor.load_state_dict(best_model_state) # for td3-bc I imagine we need both models!!!!!! SO FIX!!!!!!
    

def outer_training_loop(dataset, config_manager): 
    """Runs single outer fold - designed for HPC job array."""
    
    logger.info("Starting outer training loop...")

    # Get CV configuration
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    target_fold = config_manager.cv_config['fold']  # This comes from job array index when running on HPC
    
    if target_fold is None:
        raise ValueError("target_fold must be specified")
    
    logger.info(f"Processing outer fold {target_fold+1}/{n_outer_folds}")
    
    # Outer CV split
    kf_outer = KFold(n_splits=n_outer_folds, shuffle=True, 
                     random_state=config_manager.data_config["seed"])

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if outer_fold != target_fold:
            continue  # Skip other folds
            
        logger.info(f"=== OUTER FOLD {outer_fold} ===")
        
        # Create outer train/test splits
        train_dataset = dataset.create_subset(dataset.encounter_ids()[outer_train_id])     
        test_dataset = dataset.create_subset(dataset.encounter_ids()[outer_test_id])

        ###################   HYPERPARAMETER OPTIMIZATION #############################
        logger.info("Phase 1: Hyperparameter Optimization")
        best_hyperparameters, training_scores, val_scores, n_epochs = optimize_hyperparameters(
            train_dataset, config_manager, n_inner_folds=n_inner_folds, 
            outer_fold=outer_fold
        )
        logger.info(f"Best hyperparameters: {best_hyperparameters}")

        # get the max number of epochs used in inner folds
        n_epochs = max(n_epochs) # for final training

        #################### RETRAIN FINAL MODEL #############################
        logger.info("Phase 2: Final Model Training & Evaluation")
        
        # create train and test processors with shared normalization
        train_processor, test_processor, norm_params = create_processors(
                dataset, outer_train_id, outer_test_id, config_manager
            )        
                        
        final_training_scores, test_scores, n_epochs = train_with_hyperparameters(
                                                                train_processor, 
                                                                test_processor, 
                                                                config_manager, 
                                                                best_hyperparameters,
                                                                n_epochs=n_epochs,
                                                                final_model=True
                                                            )
        
        # PLOT THE TRAINING AND VALIDATION LOSSES
        plot_losses(training_scores, val_scores, test_scores, outer_fold, config_manager)

        
        # save the configuration_manager
        with open(config_manager.fold_dir / "config_manager.pkl", "wb") as f:
            pickle.dump(config_manager, f)  

        # create a dictionary to save the scores
        all_scores = {
            "inner_training_scores": training_scores,
            "inner_val_scores": val_scores,
            "final_training_scores": final_training_scores,
            "test_scores": test_scores,
            "best_hyperparameters": best_hyperparameters,
            "n_epochs": n_epochs
        }
        with open(config_manager.fold_dir / "all_scores.pkl", "wb") as f:
            pickle.dump(all_scores, f)

        # save the normalization parameters as a pkl file
        with open(config_manager.fold_dir / "norm_params.pkl", "wb") as f:
            pickle.dump(norm_params, f)        
            
        # Save results for this outer fold
        results_path = Path(config_manager.experiment_dir) / f"fold_{outer_fold}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'outer_fold': outer_fold,
                'best_hyperparameters': best_hyperparameters,
                'final_metrics': test_scores,
                'config': config_manager.training_config
            }, f, indent=2)
        
        logger.info(f"✓ Outer fold {outer_fold} complete. Results saved to {results_path}")
        
        return test_scores
    

def optimize_hyperparameters(train_dataset, config_manager,
                                        n_inner_folds=None, 
                                            norm_params=None, 
                                                outer_fold=None):
    """Inner CV loop for hyperparameter optimization."""
    
    # loss_metric = config_manager.training_config['loss_metric'] # e.g. actor loss, critic loss etc, depends on algorithm

    ################################ create hyperparameter space ################################
    # the hyperparameter space to search consists of the parameters that are lists 
    # in config_manager.training_config
    hyperparameter_space = {k: v for k, v in 
            config_manager.training_config.items() if isinstance(v, list)}
    if not hyperparameter_space:
        raise ValueError("No hyperparameters to optimize. Ensure that config_manager.training_config contains lists of values to search over.")
    
    # and also continuous distributions specified as strings
    for k, v in config_manager.training_config.items():
        if isinstance(v, str) and '(' in v and ')' in v:
            hyperparameter_space[k] = parse_distribution(v)

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
            inner_training_scores[inner_fold], inner_val_scores[inner_fold], inner_n_epochs[inner_fold] = \
                train_with_hyperparameters(
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

    batch_size = hyperparams['batch_size']
    loss_metric = config_manager.training_config['loss_metric']

    # Create replay buffers
    training_buffer = create_buffer(train_processor, batch_size)
    val_test_buffer = create_buffer(val_test_processor, batch_size)
    
    # create trainer
    trainer = create_trainer(train_processor, hyperparams, config_manager)
        
    # Early stopping configuration 
    early_stopping = get_early_stopping_config(config_manager)

    best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training with early stopping (patience={early_stopping['patience']}, eval_freq={early_stopping['eval_freq']}, min_delta={early_stopping['min_delta']})")
    
    # Use defaultdict to automatically create new keys
    training_losses = defaultdict(list) 
    
    if not final_model:
        val_losses = defaultdict(list)
    
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

        #################### Evaluate during hyperparameter tuning ONLY ####################
        if not final_model:            
            # Evaluate periodically
            if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                    or epoch == n_epochs - 1:
                evaluate_during_training(trainer, epoch, val_test_buffer, val_losses, config_manager)
                
                # Determine validation loss for early stopping
                current_val_loss = val_losses[loss_metric][-1]            
                logger.info(f"Epoch {epoch}: Train {loss_metric}={training_losses[loss_metric][-1]:.4f}, Val {loss_metric}={current_val_loss:.4f}")
                                
                
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

        return training_losses, val_losses, n_epochs

    ########################## evaluate on test set after training if final model ##########################
    # note that results df will be None if no targets/predictions were collected during evaluation (i.e.Q learning)
    test_losses, results_df = \
            evaluate_final_model(
            trainer, 
            val_test_buffer, 
            val_test_processor, 
            config_manager
        )
    
    logger.info(f"Final test metrics: {test_losses}")

    # Save final model if this is the final training
    model_path = Path(config_manager.fold_dir /  "final_model.pt")
    torch.save(trainer.state_dict(), model_path)
    logger.info(f" Final model saved to {model_path}")

    # Save test predictions if available
    if results_df is not None:
        results_path = Path(config_manager.fold_dir / "test_predictions.parquet")
        results_df.to_parquet(results_path)
        logger.info(f" Test predictions saved to {results_path}\n")       

    return training_losses, test_losses, n_epochs

