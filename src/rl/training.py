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
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from .evaluation import calculate_metrics
# from .plotting import create_all_plots, create_aggregated_shap_plots
from ..utils.logging_utilities import wandb_init
from ..data.data_processor import DataProcessor
# from .dataset import Dataset
from .replay_buffer import ReplayBuffer, RLDataset
from .models.behavioural_cloning import keep_best_trajectories, keep_positive_trajectories
from .models.td3_bc import TD3_BC
from ..utils.utilities import set_seed
from .plotting import plot_losses, plot_Q_by_mortality

import logging
logger = logging.getLogger(__name__)


MODEL_MODULES = {
    "any_percent_bc": "src.rl.models.behavioural_cloning",
    "basic_q": "src.rl.models.basic_q",
    "td3_bc": "src.rl.models.td3_bc"
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


def select_best_trajectories(processor, config_manager):
    """Keeps only the best trajectories based on config_manager.frac."""
    if config_manager.frac <= 0. or config_manager.frac > 1.:
        raise ValueError("config_manager.frac must be in (0, 1]")
    
    if config_manager.frac != 0.:
        keep_best_trajectories(processor, config_manager)
    else:
        keep_positive_trajectories(processor)


def create_buffer(processor, batch_size):
    dataset = RLDataset(processor.rl_dataset)
    buffer = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return buffer


def create_trainer(train_processor, hyperparams, config_manager):

    state_dim = train_processor.rl_dataset["observations"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    max_action = np.max(np.abs(train_processor.rl_dataset["actions"]))

    model_type = config_manager.model_type
    module = importlib.import_module(MODEL_MODULES[model_type])
    
    return module.get_trainer(
        config_manager,
        state_dim,
        action_dim,
        max_action,
        hyperparams
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

def estimate_v0(trainer, starting_states, batch_size, losses_dict):
    # we store the estimates in losses_dict
    # optionally, estimate V0 and append to training losses
        v0_mean, v0_sem = trainer.estimate_v0(starting_states, batch_size = batch_size)
        # print(f"Epoch {epoch}: Train mean start state values: {train_v0_mean:.4f} ± {train_v0_sem:.4f}")
        losses_dict['v0_mean'].append(float(v0_mean))
        losses_dict['v0_sem'].append(float(v0_sem))


def evaluate_during_training(trainer, epoch, val_test_buffer, batch_size, val_losses):
    
    epoch_val_losses = defaultdict(list)

    for batch_idx, batch in enumerate(val_test_buffer):      
        if batch_idx * batch_size >= 5000:  # Limit validation to 5000 samples for speed, make this a variable
            break
       
        batch_losses, _, _ = trainer.evaluate(batch)
        
        # Automatically append all validation metrics
        for key, value in batch_losses.items():
            epoch_val_losses[key].append(value)
                     
    # Calculate average validation losses for this epoch
    epoch_val_avg = {key: np.mean(values) for key, values in epoch_val_losses.items()}
    if len(epoch_val_avg) > 1:
        logger.info(f"Epoch {epoch}: Validation metrics: {epoch_val_avg}")

    # add a key for epoch number
    epoch_val_avg['epoch'] = epoch

    # Append to val_losses with epoch as key
    for key, value in epoch_val_avg.items():
        val_losses[key].append(value)  


def evaluate_final_model(trainer, test_buffer, test_processor, config_manager):
    test_losses = defaultdict(list)
    all_targets = []
    all_predictions = []

    # Final evaluation on test set
    for batch_idx, batch in enumerate(test_buffer):
        
        batch_losses, targets, predictions = trainer.evaluate(batch)

        # Automatically append all validation metrics
        for key, value in batch_losses.items():
            test_losses[key].append(value)

        if targets is not None and predictions is not None:
            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())
        
    # Calculate average test losses across all batches
    test_losses = {key: np.mean(values) for key, values in test_losses.items()}
    
    ##### make a dataframe of targets and predictions ####
    # columns are config_manager.data_config['action_cols']
    if not all_targets or not all_predictions:
        logger.info("No targets or predictions were collected during final evaluation.")
        return test_losses, None

    else:
        action_cols = test_processor.actions

        pred_cols = [col + "_pred" for col in action_cols]

        # create dataframe using all_targets and all_predictions
        all_targets = np.vstack(all_targets)
        all_predictions = np.vstack(all_predictions)

        results_df = pd.DataFrame(
            data=np.hstack([all_targets, all_predictions]),
            columns=action_cols + pred_cols
        )

        results_path = Path(config_manager.fold_dir / "test_predictions.parquet")
        results_df.to_parquet(results_path)
        logger.info(f" Test predictions saved to {results_path}\n")       

    return test_losses, results_df



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
    logger.info("Discount set to: " + str(config_manager.training_config['discount']))

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

        # Keep only best trajectories if a fraction to keep is specified
        if config_manager.model_type == "any_percent_bc":
            logger.info(f"    Selecting best trajectories (frac={config_manager.frac})")
            select_best_trajectories(train_processor, config_manager)
            select_best_trajectories(test_processor, config_manager)
                        
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

        # if "survival_Q" in test_scores, plot it
        if "survival_Q" in test_scores:
            plot_Q_by_mortality(val_scores, test_scores, outer_fold, config_manager)

        
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
    
    loss_metric = config_manager.training_config['loss_metric'] # e.g. actor loss, critic loss etc, depends on algorithm

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
        inner_training_scores = [None] * n_inner_folds
        inner_val_scores = [None] * n_inner_folds
        inner_n_epochs = [None] * n_inner_folds        

        ################# Inner CV evaluation for testing hyperparameters #################       
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True, 
                         random_state=config_manager.data_config["seed"])        
        
        for inner_fold, (inner_train_id, inner_val_id) in enumerate(kf_inner.split(train_dataset.encounter_ids())):
            logger.info(f"  Inner fold {inner_fold+1}/{n_inner_folds}")
            
            # create train and val processors with shared normalization
            inner_train_processor, inner_val_processor, _ = create_processors(
                train_dataset, inner_train_id, inner_val_id, config_manager
            )
            
            # Keep only best trajectories if a fraction to keep is specified
            if config_manager.model_type == "any_percent_bc":
                logger.info(f"    Selecting best trajectories (frac={config_manager.frac})")
                select_best_trajectories(inner_train_processor, config_manager)
                select_best_trajectories(inner_val_processor, config_manager)
           
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
    
    # get starting state distribution for V0 estimation if we're doing TD3-BC
    if config_manager.training_config['monitor_v0']:
        training_starting_states = train_processor.get_start_states()
        val_starting_states = val_test_processor.get_start_states()

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
            batch_losses = trainer.train(batch)

            # Automatically append all metrics - creates new keys if needed
            for key, value in batch_losses.items():
                epoch_training_losses[key].append(value)

        # Calculate average loss for this epoch and append to training_losses
        append_epoch_losses(epoch_training_losses, training_losses)

        # optionally, estimate V0 and append to training losses
        if config_manager.training_config['monitor_v0']:
            estimate_v0(trainer, training_starting_states, batch_size, training_losses)

        #################### Evaluate during hyperparameter tuning ONLY ####################
        if not final_model:            
            # Evaluate periodically
            if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                    or epoch == n_epochs - 1:
                evaluate_during_training(trainer, epoch, val_test_buffer, batch_size, val_losses)
                
                # Determine validation loss for early stopping
                current_val_loss = val_losses[loss_metric][-1]
                logger.info(f"Epoch {epoch}: Train {loss_metric}={training_losses[loss_metric][-1]:.4f}, Val {loss_metric}={current_val_loss:.4f}")
                                
                # optionally, estimate V0 and append to validation losses
                if config_manager.training_config['monitor_v0']:
                    estimate_v0(trainer, val_starting_states, batch_size, val_losses)
                    logger.info(f"Epoch {epoch}: Val mean start state values: {val_losses['v0_mean'][-1]:.4f} ± {val_losses['v0_sem'][-1]:.4f}")                 
                
                
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
    
    if config_manager.training_config['monitor_v0']:        
        estimate_v0(trainer, val_starting_states, batch_size, test_losses)
        
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

