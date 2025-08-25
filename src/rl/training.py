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
import torch.nn.functional as F
# from .evaluation import calculate_metrics
# from .plotting import create_all_plots, create_aggregated_shap_plots
from ..utils.logging_utilities import wandb_init
from ..data.data_processor import DataProcessor
# from .dataset import Dataset
from .replay_buffer import ReplayBuffer
from .models.behavioural_cloning import Actor, MixedActionActor, BC
from ..utils.utilities import set_seed



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


def outer_training_loop_V1(dataset, config_manager): 

    # Get CV configuration from config
    n_outer_folds = config_manager.cv_config["n_outer_folds"]
    n_inner_folds = config_manager.cv_config["n_inner_folds"]
    
    print(f"Using {n_outer_folds} outer folds and {n_inner_folds} inner folds")    

    print(f"Number of unique encounter IDs: {len(dataset.encounter_ids())}")
    
    ######################################### OUTER SPLITS #################################################
       
    # the outer test splits are held out for the final prediction
    # the outer train splits are used for hyperparameter tuning 
    kf_outer = KFold(n_splits=n_outer_folds, 
                     shuffle=True, random_state=config_manager.data_config["seed"])

    target_fold = config_manager.cv_config['fold']

    for outer_fold, (outer_train_id, outer_test_id) in enumerate(kf_outer.split(dataset.encounter_ids())):
        if target_fold is not None and outer_fold != target_fold:
            print(f"Skipping outer fold {outer_fold} as it is not the target fold {target_fold}")
            continue

        train_dataset = dataset.create_subset(dataset.encounter_ids()[outer_train_id])     
        test_dataset = dataset.create_subset(dataset.encounter_ids()[outer_test_id])
               
        # even though the train dataset will be further split for hyperparameter tuning,
        # we need to create a processor now to get normalization stats
        train_processor = DataProcessor(train_dataset, config_manager.training_config, 
                                        normalize_obs = False, normalize_actions=False,
                                        transform_actions = False, create_binary_action = False)
        keep_best_trajectories(train_processor)
        train_processor.normalize()     
        
        # we create the test processor and replay buffer, but it is used later
        test_processor = DataProcessor(test_dataset, config_manager.training_config, 
                                       norm_params=train_processor.get_normalization_params())
        keep_best_trajectories(test_processor)
        test_buffer = ReplayBuffer(test_processor.rl_dataset)

        ######################################### HYPERPARAMETER OPTIMIZATION #################################################


        # split the train dataset into train and validation sets
        kf_inner = KFold(n_splits=n_inner_folds, 
                shuffle=True, random_state=config_manager.data_config["seed"])
        
        for inner_fold, (inner_train_id, inner_val_id) in enumerate(kf_inner.split(train_dataset.encounter_ids())): 
            if target_fold is not None and inner_fold != target_fold: # for testing only
                print(f"Skipping inner fold {inner_fold} as it is not the target fold {target_fold}; FOR TESTING ONLY")
                continue

            # Create inner train and validation datasets
            inner_train_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_train_id])
            inner_val_dataset = train_dataset.create_subset(train_dataset.encounter_ids()[inner_val_id])

            # create data_processors from the datasets, and then replay buffers
            inner_train_processor = DataProcessor(inner_train_dataset, config_manager.training_config, 
                                                  norm_params=train_processor.get_normalization_params())
            inner_val_processor = DataProcessor(inner_val_dataset, config_manager.training_config, 
                                                norm_params=train_processor.get_normalization_params())


            ######### MAKE A VAL PROCESSOR, HAS TO BE NORMALIZED USING THE TRAIN PROCESSOR STATS #########
            keep_best_trajectories(inner_train_processor)
            keep_best_trajectories(inner_val_processor)

            training_buffer = ReplayBuffer(inner_train_processor.rl_dataset)
            val_replay_buffer = ReplayBuffer(inner_val_processor.rl_dataset)

            # get the max action from the dataset
            max_action = np.max(np.abs(inner_train_processor.rl_dataset["actions"])) # not sure if this should be n-dimensional???

            # set seeds
            set_seed(config_manager.seed)

            # create the actor
            state_dim = inner_train_processor.rl_dataset["observations"].shape[1]
            action_dim = inner_train_processor.rl_dataset["actions"].shape[1]
            actor = Actor(state_dim, action_dim, max_action).to("cuda" if torch.cuda.is_available() else "cpu")
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

            # the training class requires a dictionary of keyword arguments
            kwargs = {
                "max_action": max_action,
                "actor": actor,
                "actor_optimizer": actor_optimizer,
                "discount": config_manager.training_config['discount'],
                "device": config_manager.training_config['device'],
            }

            print("---------------------------------------")
            print(f"Training BC, Hosp: {config_manager.hospital}, Seed: {config_manager.seed}")
            print("---------------------------------------")

            trainer = BC(**kwargs)

            # CORL provides an option to load a checkpoint here, a path is provided
            # in config_manager.training_config

            wandb_init(config_manager.training_config)

            best_val_loss = float("inf")
            patience = 5  # Number of iterations to wait before stopping
            patience_counter = 0

            num_batches = training_buffer.size // config_manager.training_config['batch_size']

            for e in range(config_manager.training_config['n_epochs']): # previously max_timesteps       
                for batch_idx in range(num_batches):
                    if e % 100 == 0 and batch_idx == 0:
                        # print(f"Epoch {e}")

                        batch = training_buffer.sample(config_manager.training_config['batch_size'])

                        train_metrics = trainer.train(batch)
                        
                        wandb.log(train_metrics, step=trainer.total_it)
                        
                    if e % config_manager.training_config['eval_freq'] == 0:
                        # val_batch = val_replay_buffer.sample(config_manager.training_config['batch_size'])
                        # with torch.no_grad():
                        #     val_loss = F.mse_loss(trainer.actor(val_batch[0]), val_batch[1]).item()
                        # wandb.log({"val_loss": val_loss}, step=trainer.total_it)
                        # print(f"Epoch {e}, Validation Loss: {val_loss:.4f}")                      
                        
                        val_metrics = trainer.evaluate(val_replay_buffer, batch_size=512)
                
                        # Print progress
                        print(f"Epoch {e}/{config_manager.training_config['n_epochs']}")
                        print(f"  Train Loss: {train_metrics['actor_loss']:.4f} "
                            f"(Dose: {train_metrics['dose_loss']:.4f}, "
                            f"Binary: {train_metrics['binary_loss']:.4f})")
                        print(f"  Val Loss: {val_metrics['eval/total_loss']:.4f} "
                            f"(Dose RMSE: {val_metrics['eval/dose_rmse']:.4f}, "
                            f"Binary Acc: {val_metrics['eval/binary_accuracy']:.3f})")
                        
                        # log to wandb
                        wandb.log(val_metrics, step=trainer.total_it)
                              
                        # Early stopping logic
                        if val_metrics['eval/total_loss'] < best_val_loss:
                            best_val_loss = val_metrics['eval/total_loss']
                            patience_counter = 0
                            # Optionally save the best model
                            save_path = Path(config_manager.training_config['checkpoints_path']) / f"best_model_inner_fold{inner_fold}.pt"
                            torch.save(trainer.state_dict(), save_path)

                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"Early stopping at epoch {e} due to no improvement in validation loss.")
                                break
                        
                        if config_manager.training_config['checkpoints_path'] is not None:
                            torch.save(
                                trainer.state_dict(),
                                os.path.join(config_manager.training_config['checkpoints_path'], f"inner_fold{inner_fold}_checkpoint_{e}.pt"),
                            )

        # Final evaluation on the outer test set
        print("Evaluating on the outer test set...")
        
        test_metrics = trainer.evaluate(test_buffer, batch_size=512)
        print(f"Final Test Metrics: {test_metrics}")
        wandb.log(test_metrics)

        # Save the final model
        final_save_path = Path(config_manager.training_config['checkpoints_path']) / "inner_fold{inner_fold}_final_model.pt"
        torch.save(trainer.state_dict(), final_save_path)
        print(f"Final model saved to {final_save_path}")
        
   
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

        # even though the train dataset will be further split for hyperparameter tuning,
        # we need to create a processor now to get normalization stats
        train_processor = DataProcessor(train_dataset, config_manager.training_config, 
                                        normalize_obs = False, normalize_actions=False,
                                        transform_actions = False, create_binary_actions = False)
        keep_best_trajectories(train_processor)
        train_processor.normalize()     
        norm_params = train_processor.get_normalization_params() 

        ###################   HYPERPARAMETER OPTIMIZATION #############################
        print("Phase 1: Hyperparameter Optimization")
        best_hyperparameters = optimize_hyperparameters(
            train_dataset, config_manager, n_inner_folds=n_inner_folds, 
            norm_params=norm_params, outer_fold=outer_fold
        )
        print(f"Best hyperparameters: {best_hyperparameters}")

        #################### RETRAIN FINAL MODEL #############################
        print("Phase 2: Final Model Training & Evaluation")
        
        test_processor = DataProcessor(test_dataset, config_manager.training_config, 
                                       norm_params=norm_params)
        keep_best_trajectories(test_processor)
       
        final_results = train_final_model(
            train_processor, test_processor, config_manager, best_hyperparameters
        )
        
        # Save results for this outer fold
        results_path = Path(config_manager.training_config['checkpoints_path']) / f"fold_{outer_fold}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'outer_fold': outer_fold,
                'best_hyperparameters': best_hyperparameters,
                'final_metrics': final_results,
                'config': asdict(config_manager.training_config)
            }, f, indent=2)
        
        print(f"✓ Outer fold {outer_fold} complete. Results saved to {results_path}")
        return final_results
    

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
            inner_train_processor = DataProcessor(inner_train_dataset, config_manager.training_config, 
                                                  norm_params=norm_params)
            inner_val_processor = DataProcessor(inner_val_dataset, config_manager.training_config, 
                                               norm_params=norm_params)
            
            keep_best_trajectories(inner_train_processor)
            keep_best_trajectories(inner_val_processor)
            
            # Train with current hyperparameters
            val_score = train_with_hyperparameters(
                inner_train_processor, inner_val_processor, config_manager, current_params
            )
            val_scores.append(val_score)
            print(f"  Inner fold {inner_fold+1} score: {val_score:.4f}")
        
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


def train_with_hyperparameters(train_processor, val_processor, config_manager, hyperparams):
    """Train model with specific hyperparameters and return validation score."""

    # Create replay buffers
    training_buffer = ReplayBuffer(train_processor.rl_dataset)
    val_replay_buffer = ReplayBuffer(val_processor.rl_dataset)
    
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
    batch_size = hyperparams['batch_size']
    patience = config_manager.training_config['patience']  # How many evaluations to wait
    eval_freq = config_manager.training_config['eval_freq']  # Evaluate every N epochs
    min_delta = config_manager.training_config['min_delta']  # Minimum improvement

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Training with early stopping (patience={patience}, eval_freq={eval_freq})")
    
    # Training loop with early stopping for this hyperparameter combination
    for epoch in range(n_epochs):
        batch = training_buffer.sample(batch_size)
        train_metrics = trainer.train(batch)

        # Evaluate periodically
        if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                    or epoch == n_epochs - 1:
            val_metrics = trainer.evaluate(val_replay_buffer, batch_size=512)
            current_val_loss = val_metrics['eval/total_loss']

            # Check for improvement
            improvement = best_val_loss - current_val_loss
            if improvement > min_delta:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Save best model state for this hyperparameter combination
                best_model_state = trainer.actor.state_dict().copy()
                print(f"      Epoch {epoch}: New best val_loss = {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"      Epoch {epoch}: No improvement ({patience_counter}/{patience})")
            
            #  Early stopping - stop training THIS hyperparameter combination
            if patience_counter >= patience:
                print(f" Early stopping this hyperparameter combination at epoch {epoch}/{n_epochs}")
                print(f" Best validation loss: {best_val_loss:.4f}")
                break
    
    # Restore best model state before returning score
    if best_model_state is not None:
        trainer.actor.load_state_dict(best_model_state)
        print(f"    Restored best model (val_loss={best_val_loss:.4f})")

    return best_val_loss


def train_final_model(train_dataset, test_dataset, config_manager, best_hyperparams):
    """Train final model with best hyperparameters on full outer train set."""
    
    print("Training final model with best hyperparameters...")
    
    # Create processors for final training
    train_processor = DataProcessor(train_dataset, config_manager.training_config, 
                                    normalize_obs=False, normalize_actions=False,
                                    transform_actions=False, create_binary_actions=False)
    keep_best_trajectories(train_processor)
    train_processor.normalize()
    
    test_processor = DataProcessor(test_dataset, config_manager.training_config, 
                                   norm_params=train_processor.get_normalization_params())
    keep_best_trajectories(test_processor)
    
    # Create replay buffers
    training_buffer = ReplayBuffer(train_processor.rl_dataset)
    test_buffer = ReplayBuffer(test_processor.rl_dataset)
    
    # Create final model with best hyperparameters
    state_dim = train_processor.rl_dataset["observations"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    max_action = np.max(np.abs(train_processor.rl_dataset["actions"]))
    learning_rate = float(best_hyperparams['learning_rate'])
    
    
    ################# FIX TO MULTI ACTOR #################
    actor = Actor(state_dim, action_dim, max_action).to(config_manager.training_config['device'])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    
    final_trainer = BC(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        discount=config_manager.training_config['discount'],
        device=config_manager.training_config['device'],
    )
    
    # Train final model
    n_epochs = best_hyperparams['n_epochs']
    batch_size = best_hyperparams['batch_size']
    
    for epoch in range(n_epochs):
        batch = training_buffer.sample(batch_size)
        train_metrics = final_trainer.train(batch)
        
        if epoch % 100 == 0:
            print(f"Final training epoch {epoch}: loss = {train_metrics['actor_loss']:.4f}")
    
    # Final evaluation on test set
    test_metrics = final_trainer.evaluate(test_buffer, batch_size=1024)
    print(f"Final test metrics: {test_metrics}")
    
    # Save final model
    model_path = Path(config_manager.training_config['checkpoints_path']) / "final_model.pt"
    torch.save(final_trainer.state_dict(), model_path)
    
    return test_metrics