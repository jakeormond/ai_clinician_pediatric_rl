import os
import yaml
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import json
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
from .models.behavioural_cloning import Actor, MixedActionActor, BC
from ..utils.utilities import set_seed
from .plotting import plot_losses



def keep_best_trajectories(dataprocessor):
   
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    for i, (reward, done) in enumerate(zip(dataprocessor.rl_dataset["rewards"], 
                                           dataprocessor.rl_dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= dataprocessor.discount     
        
        if done == 1.0:
            if len(cur_ids) <= dataprocessor.max_traj_len:
                # don't use trajectories that are too long
                ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
            
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(dataprocessor.frac * len(sort_ord)))]
    top_returns = np.array(returns)[top_trajs]
    print(f"mean top returns: {np.mean(top_returns)}")

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataprocessor.rl_dataset["observations"] = dataprocessor.rl_dataset["observations"][order]
    dataprocessor.rl_dataset["actions"] = dataprocessor.rl_dataset["actions"][order]
    dataprocessor.rl_dataset["next_observations"] = dataprocessor.rl_dataset["next_observations"][order]
    dataprocessor.rl_dataset["rewards"] = dataprocessor.rl_dataset["rewards"][order]
    dataprocessor.rl_dataset["terminated"] = dataprocessor.rl_dataset["terminated"][order]

    return


def keep_positive_trajectories(dataprocessor):
    """
    This is for the behavioural cloning using terminal rewards only. In
    this case, it's simply a matter of keeping only survival trajectories;
    non-survival trajectories are discarded.
    """
   
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    for i, (reward, done) in enumerate(zip(dataprocessor.rl_dataset["rewards"], 
                                           dataprocessor.rl_dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= dataprocessor.discount     
        
        if done == 1.0:
            if reward < 0:
                cur_ids = []
                continue
            
            elif reward > 0:
                if len(cur_ids) <= dataprocessor.max_traj_len:
                    # don't use trajectories that are too long
                    ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
            
            cur_ids = []
            
    order = [item for sublist in ids_by_trajectories for item in sublist]    
    
    dataprocessor.rl_dataset["observations"] = dataprocessor.rl_dataset["observations"][order]
    dataprocessor.rl_dataset["actions"] = dataprocessor.rl_dataset["actions"][order]
    dataprocessor.rl_dataset["next_observations"] = dataprocessor.rl_dataset["next_observations"][order]
    dataprocessor.rl_dataset["rewards"] = dataprocessor.rl_dataset["rewards"][order]
    dataprocessor.rl_dataset["terminated"] = dataprocessor.rl_dataset["terminated"][order]

    return
        
   
def outer_training_loop(dataset, config_manager): 
    """Runs single outer fold - designed for HPC job array."""
    
    # Get CV configuration
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    target_fold = config_manager.cv_config['fold']  # This comes from job array index when running on HPC
    
    if target_fold is None:
        raise ValueError("target_fold must be specified")
    
    print(f"Processing outer fold {target_fold}/{n_outer_folds}")
    
    # Outer CV split
    kf_outer = KFold(n_splits=n_outer_folds, shuffle=True, 
                     random_state=config_manager.data_config["seed"])

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if outer_fold != target_fold:
            continue  # Skip other folds
            
        print(f"=== OUTER FOLD {outer_fold} ===")
        
        # Create outer train/test splits
        train_dataset = dataset.create_subset(dataset.encounter_ids()[outer_train_id])     
        test_dataset = dataset.create_subset(dataset.encounter_ids()[outer_test_id])

        ###################   HYPERPARAMETER OPTIMIZATION #############################
        print("Phase 1: Hyperparameter Optimization")
        best_hyperparameters = optimize_hyperparameters(
            train_dataset, config_manager, n_inner_folds=n_inner_folds, 
            outer_fold=outer_fold
        )
        print(f"Best hyperparameters: {best_hyperparameters}")

        #################### RETRAIN FINAL MODEL #############################
        print("Phase 2: Final Model Training & Evaluation")
        
        # even though the train dataset will be further split for hyperparameter tuning,
        # we need to create a processor now to get normalization stats
        train_processor = DataProcessor(train_dataset, config_manager.training_config)
        norm_params = train_processor.get_normalization_params()
        test_processor = DataProcessor(test_dataset, config_manager.training_config, 
                                            norm_params=norm_params)

        # Keep only best trajectories if a fraction to keep is specified
        if config_manager.training_config['frac'] != 0.:
            keep_best_trajectories(train_processor)
            keep_best_trajectories(test_processor)
        else:
            keep_positive_trajectories(train_processor)
            keep_positive_trajectories(test_processor)
                         
        best_val_loss, training_losses, val_losses, test_metrics = train_with_hyperparameters(
                train_processor, 
                test_processor, 
                config_manager, 
                best_hyperparameters,
                final_model=True
            )
        
        # PLOT THE TRAINING AND VALIDATION LOSSES
        plot_losses(training_losses, val_losses, outer_fold, config_manager)
        
        # Save results for this outer fold
        results_path = Path(config_manager.experiment_dir) / f"fold_{outer_fold}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'outer_fold': outer_fold,
                'best_hyperparameters': best_hyperparameters,
                'final_metrics': test_metrics,
                'config': config_manager.training_config
            }, f, indent=2)
        
        print(f"✓ Outer fold {outer_fold} complete. Results saved to {results_path}")
        return test_metrics
    

def optimize_hyperparameters(train_dataset, config_manager,
                                        n_inner_folds=None, 
                                            norm_params=None, 
                                                outer_fold=None):
    """Inner CV loop for hyperparameter optimization."""
    
    # the hyperparameter space to search consists of the parameters that are lists 
    # in config_manager.training_config
    hyperparameter_space = {k: v for k, v in 
            config_manager.training_config.items() if isinstance(v, list)}
    if not hyperparameter_space:
        raise ValueError("No hyperparameters to optimize. Ensure that config_manager.training_config contains lists of values to search over.")
    
    best_hyperparams = None
    best_val_score = float('inf')
    
    # grid search over hyperparameters 
    param_combinations = list(itertools.product(*hyperparameter_space.values()))
    param_names = list(hyperparameter_space.keys())
        
    for i, param_values in enumerate(param_combinations):
        current_params = dict(zip(param_names, param_values))
        print(f"Testing hyperparameters {i+1}/{len(param_combinations)}: {current_params}")
        
        # Inner CV evaluation for these hyperparameters
        val_scores = []
        kf_inner = KFold(n_splits=n_inner_folds, shuffle=True, 
                         random_state=config_manager.data_config["seed"])
        
        for inner_fold, (inner_train_id, inner_val_id) in enumerate(kf_inner.split(train_dataset.encounter_ids())):
            print(f"  Inner fold {inner_fold+1}/{n_inner_folds}")
            
            # Create inner splits
            inner_train_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_train_id])
            inner_val_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_val_id])
            
            # Create processors with shared normalization
            inner_train_processor = DataProcessor(inner_train_dataset, config_manager.training_config)
            norm_params = inner_train_processor.get_normalization_params()
            inner_val_processor = DataProcessor(inner_val_dataset, config_manager.training_config, 
                                               norm_params=norm_params)
            
            if config_manager.training_config['frac'] != 0.:
                keep_best_trajectories(inner_train_processor)
                keep_best_trajectories(inner_val_processor)
            else:
                keep_positive_trajectories(inner_train_processor)
                keep_positive_trajectories(inner_val_processor)
           
            # Train with current hyperparameters
            best_val_loss, training_losses, val_losses = train_with_hyperparameters(
                inner_train_processor, inner_val_processor, config_manager, current_params
            )
            val_scores.append(best_val_loss)
            print(f"  Inner fold {inner_fold+1} score: {best_val_loss:.4f}")
        
        # Average validation score across inner folds
        avg_val_score = np.mean(val_scores)
        std_val_score = np.std(val_scores)
        print(f"Average validation score: {avg_val_score:.4f} ± {std_val_score:.4f}")
        
        # Update best hyperparameters
        if avg_val_score < best_val_score:
            improvement = best_val_score - avg_val_score
            best_val_score = avg_val_score
            best_hyperparams = current_params.copy()
            print(f"New best hyperparameters found! (improvement: {improvement:.4f})")
        else:
            print(f"No improvement (current best: {best_val_score:.4f})")

    print(f"\nHyperparameter optimization complete!")
    print(f"Best validation score: {best_val_score:.4f}")
    print(f"Best hyperparameters: {best_hyperparams}")

    return best_hyperparams


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



def train_with_hyperparameters(train_processor, val_test_processor, config_manager, hyperparams, final_model=False):
    """Train model with specific hyperparameters and return validation score."""
    
    batch_size = hyperparams['batch_size']

    # Create replay buffers
    train_dataset = RLDataset(train_processor.rl_dataset)
    training_buffer = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
        )

    val_test_dataset = RLDataset(val_test_processor.rl_dataset)
    val_test_buffer = DataLoader(
        val_test_dataset, 
        batch_size=batch_size, 
        shuffle=False
        )
    
    # Create model with hyperparameters
    state_dim = train_processor.rl_dataset["observations"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    max_action = np.max(np.abs(train_processor.rl_dataset["actions"]))    
    learning_rate = float(hyperparams['learning_rate'])

    actor = MixedActionActor(
        state_dim=state_dim, 
        dose_actions=action_dim//2, 
        binary_actions=action_dim//2, 
        hidden_dim=hyperparams['hidden_dim'], 
        max_action = max_action
        ).to(config_manager.training_config['device'])
    
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), 
        lr=learning_rate
        )
    
    trainer = BC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        discount=config_manager.training_config['discount'],
        device=config_manager.training_config['device'],
        dose_actions=action_dim//2,
        binary_actions=action_dim//2
    )
    
    # Early stopping configuration for THIS hyperparameter combination
    n_epochs = config_manager.training_config['n_epochs']
    patience = config_manager.training_config['patience']  # How many evaluations to wait
    eval_freq = config_manager.training_config['eval_freq']  # Evaluate every N epochs
    min_delta = config_manager.training_config['min_delta']  # Minimum improvement

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Training with early stopping (patience={patience}, eval_freq={eval_freq})")
    training_losses = []
    val_losses = {}
    
    # Training loop with early stopping for this hyperparameter combination
    for epoch in range(n_epochs):

        epoch_losses = []

        for batch_idx, batch in enumerate(training_buffer):
            batch_losses = trainer.train(batch)
            epoch_losses.append(batch_losses['actor_loss'])

        # Calculate average loss for this epoch
        avg_epoch_loss = np.mean(epoch_losses)
        training_losses.append(avg_epoch_loss)

        # Evaluate periodically
        if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                   or epoch == n_epochs - 1:
            current_val_loss = []
            for batch_idx, batch in enumerate(val_test_buffer):
                if batch_idx * batch_size >= 5000:  # Limit validation to 5000 samples for speed
                    break
                val_metrics = trainer.evaluate(batch)
                current_val_loss.append(val_metrics['eval/total_loss'])

            val_losses[epoch] = float(np.mean(current_val_loss))

            # Check for improvement
            improvement = best_val_loss - val_losses[epoch]
            if improvement > min_delta:
                best_val_loss = val_losses[epoch]
                patience_counter = 0
                # Save best model state for this hyperparameter combination
                best_model_state = trainer.actor.state_dict().copy()
                print(f"Epoch {epoch}: Train={avg_epoch_loss:.4f}, Val={best_val_loss:.4f} (NEW BEST)")
            else:
                patience_counter += 1                
                print(f"Epoch {epoch}: Train={avg_epoch_loss:.4f}, Val={val_losses[epoch]:.4f} ({patience_counter}/{patience} - NO IMPROVEMENT)")
                
            #  Early stopping - stop training THIS hyperparameter combination
            if patience_counter >= patience:
                
                print(f" Early stopping this hyperparameter combination at epoch {epoch}/{n_epochs}")
                print(f" Best validation loss: {best_val_loss:.4f}")
                break
    
    # Restore best model state before returning score
    if best_model_state is not None:
        trainer.actor.load_state_dict(best_model_state)
        print(f"    Restored best model (val_loss={best_val_loss:.4f})")

    
    if final_model:
        epoch_losses = []

        # Final evaluation on test set
        for batch_idx, batch in enumerate(val_test_buffer):
            batch_metrics = trainer.evaluate(batch)
            epoch_losses.append(batch_metrics)
        
        test_metrics = average_dicts(epoch_losses)       
        print(f"Final test metrics: {test_metrics}")

        # Save final model if this is the final training
        model_path = Path(config_manager.fold_dir /  "final_model.pt")
        torch.save(trainer.state_dict(), model_path)
        print(f" Final model saved to {model_path}")
   
        return best_val_loss, training_losses, val_losses, test_metrics
    
    else:
        return best_val_loss, training_losses, val_losses


# def train_final_model(train_processor, test_processor, config_manager, best_hyperparams):
#     """Train final model with best hyperparameters on full outer train set."""
    
#     print("Training final model with best hyperparameters...")

#     batch_size = best_hyperparams['batch_size']
    
#     train_dataset = RLDataset(train_processor.rl_dataset)
#     training_buffer = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True
#         )
    
#     test_dataset = RLDataset(test_processor.rl_dataset)
#     test_buffer = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False
#         )
    

#     # Create final model with best hyperparameters
#     state_dim = train_processor.rl_dataset["observations"].shape[1]
#     action_dim = train_processor.rl_dataset["actions"].shape[1]
#     max_action = np.max(np.abs(train_processor.rl_dataset["actions"]))
#     learning_rate = float(best_hyperparams['learning_rate'])
    
    
#     ################# FIX TO MULTI ACTOR #################
#     actor = Actor(state_dim, action_dim, max_action).to(config_manager.training_config['device'])
#     actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    
#     final_trainer = BC(
#         max_action=max_action,
#         actor=actor,
#         actor_optimizer=actor_optimizer,
#         discount=config_manager.training_config['discount'],
#         device=config_manager.training_config['device'],
#     )
    
#     # Train final model
#     n_epochs = best_hyperparams['n_epochs']
#     batch_size = best_hyperparams['batch_size']
    
#     for epoch in range(n_epochs):
#         batch = training_buffer.sample(batch_size)
#         train_metrics = final_trainer.train(batch)
        
#         if epoch % 10 == 0:
#             print(f"Final training epoch {epoch}: loss = {train_metrics['actor_loss']:.4f}")
    
#     # Final evaluation on test set
#     test_metrics = final_trainer.evaluate(test_buffer, batch_size=1024)
#     print(f"Final test metrics: {test_metrics}")
    
#     # Save final model
#     model_path = Path(config_manager.training_config['checkpoints_path']) / "final_model.pt"
#     torch.save(final_trainer.state_dict(), model_path)
    
#     return test_metrics