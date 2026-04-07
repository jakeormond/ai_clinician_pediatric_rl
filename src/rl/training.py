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
from ..data.data_processor_for_shards import ShardProcessor
# from .dataset import Dataset
from .replay_buffer import RLDataset
from .models.behavioural_cloning import keep_best_trajectories, keep_positive_trajectories
from ..supervised_learning.training import get_hyperparameter_space, select_hyperparameters_randomly, get_fixed_hyperparameters, to_json_safe
from .models.td3_bc import TD3_BC
from ..utils.utilities import set_seed
from .plotting import plot_losses, plot_Q_by_mortality, plot_gradient_norms


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



def create_processor_from_shards(config_manager, normalize=True, set_to_return='all'):
    """
    Copied from supervised_learning/training.py 
    """
    train_processor = ShardProcessor(config_manager, normalize=normalize, split='train')
    val_processor = ShardProcessor(config_manager, split='val',
                                  norm_params=train_processor.get_normalization_params())
    test_processor = ShardProcessor(config_manager, split='test',
                                   norm_params=train_processor.get_normalization_params())

    if set_to_return == 'all':
        return train_processor, val_processor, test_processor
    elif set_to_return == 'train':
        return train_processor
    elif set_to_return == 'val':
        return val_processor
    elif set_to_return == 'test':
        return test_processor
    

def create_buffer(processor, batch_size, shuffle=True, scaler=None, action_names=None):
    scaler = processor.create_rl_dataset(states="model_outputs", scaler=scaler, action_names=action_names) # states argument is not currently used, but we can add options for different state representations in the future
    dataset = RLDataset(processor) # model_outputs refer to dynamic_deephit transformer outputs; other options not currently implemented
    buffer = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return buffer, scaler


def create_trainer(train_processor, hyperparams, config_manager):

    state_dim = train_processor.rl_dataset["states"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    max_action = np.max(np.abs(train_processor.rl_dataset["actions"].cpu().numpy()))

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
    
    # calculate q value stats if available
    q_stats = None
    if "mean_Q" in epoch_training_losses:
        with torch.no_grad():
            q_stats, epoch_training_losses = calculate_q_stats(epoch_training_losses)
    
    epoch_avg_losses = {key: np.mean(values) for key, values in epoch_training_losses.items()}
    # Append epoch averages to training_losses
    for key, value in epoch_avg_losses.items():
        training_losses[key].append(value)     

    # add q stats if available
    if q_stats is not None:
        for key, value in q_stats.items():
            training_losses[key].append(value)

def estimate_v0(trainer, starting_states, batch_size, losses_dict):
    # we store the estimates in losses_dict
    # optionally, estimate V0 and append to training losses
        v0_mean, v0_sem = trainer.estimate_v0(starting_states, batch_size = batch_size)
        # print(f"Epoch {epoch}: Train mean start state values: {train_v0_mean:.4f} ± {train_v0_sem:.4f}")
        losses_dict['v0_mean'].append(float(v0_mean))
        losses_dict['v0_sem'].append(float(v0_sem))


def append_grad_norms(epoch_grad_norm, grad_norms):
    """Calculates average gradient norms for the epoch and appends to grad_norms."""
    epoch_avg_grad_norm = {key: np.mean(values) for key, values in epoch_grad_norm.items()}
    # Append epoch averages to grad_norms
    for key, value in epoch_avg_grad_norm.items():
        grad_norms[key].append(value)


def calculate_q_stats(losses_dict):
    # calculate q value stats if available
    mean_Qs = losses_dict["mean_Q"]
    variance_Qs = losses_dict["variance_Q"]
    counts_all = losses_dict["batch_n"]

    overall_mean_Q = np.sum(np.array(mean_Qs) * np.array(counts_all)) / np.sum(counts_all)
    overall_variance_Q = np.sum(counts_all * (variance_Qs + np.array(mean_Qs - overall_mean_Q)**2)) / np.sum(counts_all)
    overall_std_Q = np.sqrt(overall_variance_Q)

    # survival Qs
    mean_survival_Qs = losses_dict["survival_Q"]
    variance_survival_Qs = losses_dict["survival_variance_Q"]
    counts_survival = losses_dict["survival_n"]

    # drop nan values from mean_survival_Qs and variance_survival_Qs and zeros from counts_survival
    mean_survival_Qs = [q for q in mean_survival_Qs if not np.isnan(q)]
    variance_survival_Qs = [v for v in variance_survival_Qs if not np.isnan(v)]
    counts_survival = [c for c in counts_survival if c > 1]

    overall_mean_survival_Q = np.sum(np.array(mean_survival_Qs) * np.array(counts_survival)) / np.sum(counts_survival)
    overall_variance_survival_Q = np.sum(counts_survival * (variance_survival_Qs + np.array(mean_survival_Qs - overall_mean_survival_Q)**2)) / np.sum(counts_survival)
    overall_std_survival_Q = np.sqrt(overall_variance_survival_Q)

    # non-survival Qs
    mean_non_survival_Qs = losses_dict["non_survival_Q"]
    variance_non_survival_Qs = losses_dict["non_survival_variance_Q"]
    counts_non_survival = losses_dict["non_survival_n"]

    # drop nan values from mean_non_survival_Qs and variance_non_survival_Qs and zeros from counts_non_survival
    mean_non_survival_Qs = [q for q in mean_non_survival_Qs if not np.isnan(q)]
    variance_non_survival_Qs = [v for v in variance_non_survival_Qs if not np.isnan(v)]
    counts_non_survival = [c for c in counts_non_survival if c > 0]

    overall_mean_non_survival_Q = np.sum(np.array(mean_non_survival_Qs) * np.array(counts_non_survival)) / np.sum(counts_non_survival)
    overall_variance_non_survival_Q = np.sum(counts_non_survival * (variance_non_survival_Qs + np.array(mean_non_survival_Qs - overall_mean_non_survival_Q)**2)) / np.sum(counts_non_survival)
    overall_std_non_survival_Q = np.sqrt(overall_variance_non_survival_Q)

    survival_non_survival_diff = overall_mean_survival_Q - overall_mean_non_survival_Q
    survival_non_survival_snr = survival_non_survival_diff**2/(0.5*(overall_variance_survival_Q + overall_variance_non_survival_Q))

    # create a dictionary of these values 
    q_stats = {
        "mean_Q": overall_mean_Q,
        "std_Q": overall_std_Q,
        "mean_survival_Q": overall_mean_survival_Q,
        "std_survival_Q": overall_std_survival_Q,
        "mean_non_survival_Q": overall_mean_non_survival_Q,
        "std_non_survival_Q": overall_std_non_survival_Q,
        "survival_non_survival_diff": survival_non_survival_diff,
        "survival_non_survival_snr": survival_non_survival_snr
    }

    # No longer dropping, because we are tracking across iterations
    # drop q keys from losses
    # for key in ["mean_Q", "variance_Q", "batch_n",
    #             "survival_Q", "survival_variance_Q", "survival_n",
    #             "non_survival_Q", "non_survival_variance_Q", "non_survival_n"]:
    #     losses_dict.pop(key, None)


    return q_stats, losses_dict


def save_results(
        training_scores,
        val_scores,
        test_scores, 
        hyperparameters,
        n_epochs,
        config_manager
    ):

    # save the configuration_manager
    with open(config_manager.experiment_dir / "config_manager.pkl", "wb") as f:
        pickle.dump(config_manager, f)  

    # create a dictionary to save the scores
    all_scores = {
        "training_scores": training_scores,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "hyperparameters": hyperparameters,
        "n_epochs": n_epochs
    }
    with open(config_manager.experiment_dir / "all_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)

    # Save results for this outer fold
    results_path = Path(config_manager.experiment_dir) / f"results.json"

    with open(results_path, 'w') as f:
        json.dump(to_json_safe({
            'best_hyperparameters': hyperparameters,
            # 'time_bins': config_manager.time_bins,
            'final_metrics': test_scores,
            'config': config_manager.training_config
        }), f, indent=2)
    
    logger.info(f"✓ Training complete. Results saved to {results_path}")

    return all_scores




def evaluate_during_training(trainer, epoch, val_test_buffer, batch_size, val_losses):
    
    epoch_val_losses = defaultdict(list)

    for batch_idx, batch in enumerate(val_test_buffer):      
        if batch_idx * batch_size >= 50000:  # Limit validation to 5000 samples for speed, make this a variable
            break
       
        batch_losses, _, _ = trainer.evaluate(batch)
        
        # Automatically append all validation metrics
        for key, value in batch_losses.items():
            epoch_val_losses[key].append(value)
                     
    # calculate q value stats if available
    if "mean_Q" in epoch_val_losses:
        with torch.no_grad():
            q_stats, epoch_val_losses = calculate_q_stats(epoch_val_losses)

    # Calculate average validation losses for this epoch
    epoch_val_avg = {key: np.mean(values) for key, values in epoch_val_losses.items()}
    if len(epoch_val_avg) > 1:
        logger.info(f"Epoch {epoch}: Validation metrics: {epoch_val_avg}")

    # add a key for epoch number
    epoch_val_avg['epoch'] = epoch
    epoch_val_avg['total_it'] = trainer.total_it

    # add q stats if available
    if 'q_stats' in locals():
        epoch_val_avg.update(q_stats)

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
        
    # calculate q value stats if available
    if "mean_Q" in test_losses:
        with torch.no_grad():
            q_stats, test_losses = calculate_q_stats(test_losses)
    
    # Calculate average test losses across all batches
    test_losses = {key: np.mean(values) for key, values in test_losses.items()}
    
    # add q stats if available
    if 'q_stats' in locals():
        test_losses.update(q_stats)
        
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


def plot_results(training_losses, val_losses, test_losses, config_manager):
    
    plot_losses(training_losses, val_losses, test_losses, config_manager)

    if config_manager.model_type == "basic_q":
        plot_Q_by_mortality(val_losses, test_losses, config_manager)



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
    

def single_split_training(config_manager): 
    """ 
    based on supervised training.py file - at some point we might want to merge these two files and have a flag for whether to do RL or supervised training, but for now keeping separate for clarity.
    
    We need to select what input (state) data to use here - either the windowed data from the ShardProcessor,
    or the transformer output from dynamic deephit. We can specify this in the config_manager and then create the appropriate processor here.
    
    """
    logger.info("Starting single split training...")
    logger.info("Discount set to: " + str(config_manager.training_config['discount']))

    # create processors
    # NEED TO ADD CONFIGURATION OPTION TO USE THE FEATURE DATA OR THE TRANSFORMER OUTPUT FROM DYNAMIC DEEPHIT MODEL
    train_processor, val_processor, test_processor = create_processor_from_shards(
        config_manager, 
        normalize=True
        )


    # if config_manager.training_config['tuning'] and config_manager.n_search_iterations > 0:
    if config_manager.search_variant == 'random_search':
        logger.info(" Hyperparameter tuning mode enabled.")

        # hyperparameter optimization
        trainer, hyperparams, training_scores, val_scores, n_epochs, grad_norm = optimize_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager
        )

        logger.info(f"Best hyperparameters: {hyperparams}")

    else:
        logger.info(" Fixed hyperparameter training mode enabled.")

        # get fixed hyperparameters
        hyperparams = get_fixed_hyperparameters(config_manager)

        # train final model
        training_scores, val_scores, n_epochs, trainer, grad_norm, scaler = train_with_hyperparameters( # we can probably remove the final model flag when we refactor train_with_hyperparameters to not do early stopping during final training, since we won't be doing early stopping during final training
            train_processor, 
            val_processor, 
            config_manager, 
            hyperparams,
            n_epochs=config_manager.n_epochs,
            final_model=False
        )

    val_buffer, _ = create_buffer(val_processor, config_manager.training_config['batch_size'], shuffle=False, scaler=scaler, action_names=config_manager.data_config['actions'])
    final_val_losses, _ = evaluate_final_model( # this evaluates on the entire test set (during training we only use a subset for speed)
        trainer, 
        val_buffer, 
        val_processor,
        config_manager
    )

    logger.info(f"Final val metrics: {final_val_losses}")

    # Save final model if this is the final training
    model_path = Path(config_manager.experiment_dir /  "final_model.pt")
    if config_manager.model_type in ['basic_q']:
        torch.save(trainer.critic.state_dict(), model_path)
    else:
        torch.save(trainer.model.state_dict(), model_path)
    logger.info(f" Final model saved to {model_path}")

    # plot losses
    plot_results(training_scores, val_scores, final_val_losses, config_manager)

    # plot gradient norms 
    plot_gradient_norms(grad_norm, config_manager)

    # save results
    all_scores = save_results(
        training_scores,
        val_scores,
        final_val_losses,
        hyperparams,
        n_epochs,
        config_manager
    )

    return all_scores    
            

def optimize_hyperparameters(train_processor, val_processor, config_manager):
    """Inner CV loop for hyperparameter optimization."""
    
    # create hyperparameter space
    hyperparameter_space = get_hyperparameter_space(config_manager)

    # initialize best_val_score
    loss_metric = config_manager.training_config['loss_metric'] 
    if loss_metric in ['critic_loss']:
        best_val_loss = float('inf')
    else:        
        raise NotImplementedError("Currently only implemented for critic_loss, which is the default loss_metric for basic_q. For other metrics, we need to decide how to calculate an overall validation score across epochs and implement that logic here.")
    
    # Keep only best trajectories if a fraction to keep is specified
    if config_manager.model_type == "any_percent_bc":
        logger.info(f"    Selecting best trajectories (frac={config_manager.frac})")
        select_best_trajectories(train_processor, config_manager)
        select_best_trajectories(val_processor, config_manager)    
    
    ##################### loop through hyperparameter combinations #####################
    logger.info(f"Starting hyperparameter optimization with {config_manager.n_search_iterations} iterations...")
    
    for i in range(config_manager.n_search_iterations):
        ################# select hyperparameters randomly #################
        current_params = select_hyperparameters_randomly(
            hyperparameter_space, config_manager
        )       
        logger.info(f"Testing hyperparameters {i+1}/{config_manager.n_search_iterations}: {current_params}")

        ################# Train with current hyperparameters #################       
        training_scores, val_scores, n_epochs, trainer, grad_norm = train_with_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager, 
            current_params,
            final_model=False
        )
        
        ###################### Evaluate and update best hyperparameters ########################
        if loss_metric in ['critic_loss']:
            # critic_loss, lower is better
            current_val_loss = min(val_scores[loss_metric])
            improvement = best_val_loss - current_val_loss

        else:
            # for loss, lower is better
            # current_val_loss = min(val_scores[loss_metric])
            # improvement = best_val_loss - current_val_loss
            raise NotImplementedError("Currently only implemented for critic_loss, which is the default loss_metric for basic_q. For other metrics, we need to decide how to calculate an overall validation score across epochs and implement that logic here.")

        logger.info(f"Search iteration {i+1} validation {loss_metric}: {current_val_loss:.4f}")
        
        if improvement > 0.:
            logger.info(f" New best hyperparameters found! (improvement: {improvement:.4f})\n")
            logger.info(f"Best hyperparameters after {i+1} iterations: {current_params}")
            best_trainer = trainer
            best_val_loss = current_val_loss
            best_hyperparams = current_params.copy()
            training_scores_best = training_scores
            val_scores_best = val_scores
            grad_norm_best = grad_norm
            n_epochs_best = n_epochs
        else:
            logger.info(f"No improvement (current best: {best_val_loss:.4f})\n")

    ############################ Summarize hyperparameter optimization results #############################
    logger.info(f"Best hyperparameters after {config_manager.n_search_iterations} iterations: {best_hyperparams}")
    logger.info(f"\n                 Hyperparameter optimization complete!")

    return best_trainer, best_hyperparams, training_scores_best, val_scores_best, n_epochs_best, grad_norm_best


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
    training_buffer, scaler = create_buffer(train_processor, batch_size, action_names=config_manager.data_config['actions'])
    val_test_buffer, _ = create_buffer(val_test_processor, batch_size, scaler=scaler, action_names=config_manager.data_config['actions'])
    
    # get total number of steps for scheduler if needed
    hyperparams['total_steps'] = n_epochs * len(training_buffer)
    
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
    grad_norms = defaultdict(list)
    
    # if not final_model:
    #    val_losses = defaultdict(list)
    val_losses = defaultdict(list)
    # evaluate before any training for epoch 0
    evaluate_during_training(trainer, -1, val_test_buffer, batch_size, val_losses)   
    current_val_loss = val_losses[loss_metric][-1]  
    logger.info(f"Epoch {-1}: Val {loss_metric}={current_val_loss:.4f}")
    
    ############################### Training loop #################################
    testing = getattr(config_manager, 'testing', False)
    if testing:
        logger.info(" Testing mode enabled: will limit training to 10 epochs and 100 batches per epoch.")
        n_epochs = min(n_epochs, 5)
    
    torch.cuda.empty_cache()

    for epoch in range(n_epochs):
        # print epoch
        logger.info(f"Epoch {epoch+1}/{n_epochs}")  
        epoch_training_losses = defaultdict(list)
        epoch_grad_norm = defaultdict(list)

        for batch_idx, batch in enumerate(training_buffer):
            batch_losses, grad_norm = trainer.train(batch)

            # Automatically append all metrics - creates new keys if needed
            for key, value in batch_losses.items():
                epoch_training_losses[key].append(value)
            
            for key, value in grad_norm.items():
                epoch_grad_norm[key].append(value)

            # check if we are simply testing the training loop, in which case we break after 100 batches
            if batch_idx >= 10 and testing:
                logger.info(" Testing mode: breaking after 10 batches.")
                break

        # step the scheduler
        if trainer.scheduler is not None and config_manager.model_type in ['basic_q']:
            trainer.scheduler.step()
            print(f"Epoch {epoch} finished. New LR: {trainer.critic_optimizer.param_groups[0]['lr']:.2e}")

        # Calculate average loss for this epoch and append to training_losses
        append_epoch_losses(epoch_training_losses, training_losses)
        append_grad_norms(epoch_grad_norm, grad_norms)
        # print grad norm here
        logger.info(f"Epoch {epoch}: Train {loss_metric}={training_losses[loss_metric][-1]:.4f}, Grad Norms: {[f'{key}={value[-1]:.4f}' for key, value in grad_norms.items()]}")
        

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

        return training_losses, val_losses, n_epochs, trainer, grad_norms, scaler

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

    return training_losses, val_losses, test_losses, grad_norms

