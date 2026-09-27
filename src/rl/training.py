import importlib
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict
import random
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import wandb
from scipy.stats import uniform, loguniform
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ..utils.logging_utilities import wandb_init, generate_run_seed, get_job_id, can_reach_wandb
from ..data.data_processor import DataProcessor
from ..data.data_processor_for_shards import ShardProcessor
from .replay_buffer import RLDataset
from .models.behavioural_cloning import keep_best_trajectories, keep_positive_trajectories
from ..supervised_learning.training import get_hyperparameter_space, select_hyperparameters_randomly, get_fixed_hyperparameters, to_json_safe, train_with_hyperparameters
from .models.td3_bc import TD3_BC
from ..utils.utilities import set_seed
from .plotting import plot_results
from .models.shared_components import TrainerUtils


import logging
logger = logging.getLogger(__name__)


MODEL_MODULES = {
    "any_percent_bc": "src.rl.models.behavioural_cloning",
    "basic_q": "src.rl.models.basic_q",
    "td3_bc": "src.rl.models.td3_bc",
    "conservative_q_learning": "src.rl.models.conservative_q_learning",
    "cql_with_logprobs": "src.rl.models.cql_with_logprobs",
    "cvp": "src.rl.models.cvp",
    "iql": "src.rl.models.iql",
    "iql_with_sarsa": "src.rl.models.iql_with_sarsa",
    "bc_with_logprobs": "src.rl.models.bc_with_logprobs",
    "fqe": "src.rl.models.basic_q",  # FQE reuses basic_q's Q_Learner entirely -- see get_trainer(..., actor=...)
}

DISTRIBUTIONS = {
    "loguniform": loguniform,
    "uniform": uniform,
}

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
    

def create_buffer(
        processor, 
        batch_size, 
        hyperparams,
        shuffle=True, 
        scaler=None, 
        action_names=None
    ):
    # should correct output to include processor as it gets modified by the call to .create_rl_dataset

    use_compressed_latents = hyperparams['use_compressed_latents']
    if hyperparams.get('n_step') is None:
        n_step = 1
    else:
        n_step = hyperparams['n_step']
    gamma = hyperparams['discount']
    actions_shift = hyperparams.get('actions_shift', 0.0)  # Default to 0.0 if not provided

    scaler = processor.create_rl_dataset(
        states="model_outputs", 
        scaler=scaler, 
        action_names=action_names, 
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma,
        actions_shift=actions_shift
    ) 
    if n_step == None or n_step <= 1:
        use_n_step = False
    else:
        use_n_step = True
    dataset = RLDataset(processor) # model_outputs refer to dynamic_deephit transformer outputs; other options not currently implemented
    buffer = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return buffer, scaler 


def create_trainer(train_processor, hyperparams, config_manager, actor=None):
    state_dim = train_processor.rl_dataset["states"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    
    model_type = config_manager.model_type
    module = importlib.import_module(MODEL_MODULES[model_type])
    
    if model_type == "fqe" and actor is None:
        raise ValueError("Actor must be provided for FQE model type.")
    
    if model_type == "fqe":
        return module.get_trainer(
            config_manager,
            state_dim,
            action_dim,
            hyperparams,
            actor=actor
        )
    else:
        return module.get_trainer(
            config_manager,
            state_dim,
            action_dim,
            hyperparams
        )


def load_actor(train_processor, hyperparams, config_manager):
    state_dim = train_processor.rl_dataset["states"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]

    actor_type = config_manager.actor_type
    module = importlib.import_module(MODEL_MODULES[actor_type])

    actor = module.load_trained_actor(
        state_dim,
        action_dim,
        config_manager=config_manager        
    ) # Assuming the module has a load_trained_actor function

    return actor


def create_evaluator(config_manager, train_processor):   
    state_dim = train_processor.rl_dataset["states"].shape[1]
    action_dim = train_processor.rl_dataset["actions"].shape[1]
    # evaluator is the sarsa critic 
    module = importlib.import_module(MODEL_MODULES['basic_q'])
    return module.get_evaluator(state_dim, action_dim, config_manager)

   
def get_early_stopping_config(config_manager):
    """Extracts early stopping configuration from config_manager."""
    early_stopping = {
        "flag": config_manager.training_config.get('early_stopping_flag', False),
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
    # if "mean_Q" in epoch_training_losses:
    #     with torch.no_grad():
    #         q_stats, epoch_training_losses = calculate_q_stats(epoch_training_losses)
    
    # epoch_avg_losses = {key: np.mean(values) for key, values in epoch_training_losses.items()}
    # Append epoch averages to training_losses
   # for key, value in epoch_avg_losses.items():
    for key, value in epoch_training_losses.items():        
        training_losses[key].append(value)     

    # add q stats if available
    if q_stats is not None:
        for key, value in q_stats.items():
            training_losses[key].append(value)
    

class TrainingMetrics:
    def __init__(self):
        self._metric_keys = [
            "batch_training_losses",
            "epoch_training_losses",
            "val_losses",
            "final_val_losses",
            "batch_training_q_stats",
            "epoch_training_q_stats",
            "val_q_stats",
            "final_val_q_stats",
            "val_advantages",
            "final_val_advantages",
            "batch_training_grad_norms",
            "epoch_training_grad_norms",
            "val_neuron_stats",
            "final_val_neuron_stats",
        ]

        # 2. Dynamically initialize them as empty defaultdicts
        for key in self._metric_keys:
            setattr(self, key, defaultdict(list))

    def append_epoch_metrics(self, **kwargs) -> None:
        """Accepts any metrics container as a keyword argument
        and dynamically appends its contents to the matching attribute.
        """
        for container_name, incoming_dict in kwargs.items():
            # Check if this is a container we manage and that it actually contains data
            if incoming_dict is None or container_name not in self._metric_keys:
                continue

            # Safety check specifically for the 'advantages' edge case
            # (Checking if the dictionary is empty or has uninitialized keys)
            if "advantages" in container_name and (
                not incoming_dict or len(list(incoming_dict.values())) == 0
            ):
                continue

            # Target the matching internal defaultdict attribute
            target_storage = getattr(self, container_name)

            # Append all values cleanly
            for key, value in incoming_dict.items():
                target_storage[key].append(value)

    def get_current_epoch_val_loss(self, loss_key):
        if self.val_losses[f"val/{loss_key}"]:
            return self.val_losses[f"val/{loss_key}"][-1]
        else:
            return None

    def get_current_epoch_train_loss(self, loss_key):
        if self.epoch_training_losses[f"train/epoch_{loss_key}"]:
            return self.epoch_training_losses[f"train/epoch_{loss_key}"][-1]
        else:
            return None

    def get_current_epoch_grad_norm(self):
        if self.epoch_training_grad_norms:
            return {key: values[-1] for key, values in self.epoch_training_grad_norms.items()}
        else:
            return None
    

def save_results(
        training_metrics,
        hyperparameters,
        n_epochs,
        run_dir,
        config_manager
    ):

    # save the configuration_manager if not already saved (it should be saved at the start of the experiment, but we can save again here to capture any updates to the config_manager during training, such as normalization parameters)
    if not (config_manager.experiment_dir / "config_manager.pkl").exists():    
        with open(config_manager.experiment_dir / "config_manager.pkl", "wb") as f:
            pickle.dump(config_manager, f)  

    # training_scores is the final value in each value in training_metrics.epoch_training_losses
    training_losses = {key: values[-1] for key, values in training_metrics.epoch_training_losses.items()}
    training_q_stats = {key: values[-1] for key, values in training_metrics.epoch_training_q_stats.items()}
    training_scores = {**training_losses, **training_q_stats}

    val_scores = {
        **training_metrics.final_val_losses, **training_metrics.final_val_q_stats, 
        **training_metrics.final_val_advantages, **training_metrics.final_val_neuron_stats,
    }
    
    # create a dictionary to save the scores
    all_scores = {
        "val_scores": val_scores,
        "training_scores": training_scores,
        "hyperparameters": hyperparameters,
        "n_epochs": n_epochs
    }
    with open(run_dir / "all_scores.pkl", "wb") as f:
        pickle.dump(all_scores, f)

    # Save results for this outer fold
    results_path = Path(run_dir) / f"results.json"

    with open(results_path, 'w') as f:
        json.dump(to_json_safe({
            'hyperparameters': hyperparameters,
            # 'time_bins': config_manager.time_bins,            
            'val_scores': val_scores,
            'training_scores': training_scores,
            'config': config_manager.training_config
        }), f, indent=2)
    
    logger.info(f"✓ Training complete. Results saved to {results_path}")

    return all_scores


def evaluate_during_training(
    trainer, 
    evaluator, 
    epoch, 
    val_buffer, 
    batch_size, 
    final=False
    ):
    
    val_losses = defaultdict(list)
    val_q_stats = defaultdict(list)
    val_neuron_stats = defaultdict(list)
    val_advantages = defaultdict(list)

    for batch_idx, batch in enumerate(val_buffer):      
        if not final:
            if batch_idx * batch_size >= 50000:  # Limit validation to 5000 samples for speed, make this a variable
                break
       
        batch_losses, q_stats, neuron_stats = trainer.evaluate(batch)
        
        # run SARSA evaluation if not basic_q model - 
        if evaluator is not None: # we set evaluator to None for basic_q model since we are not calculating advantage during training for that model, but for the other models we want to calculate advantage during training to track how the learned policy is improving compared to the clinician policy
            # get q advantage
            mortality = batch[6]  # assuming mortality is the 7th element in the batch tuple
            with torch.no_grad():
                rl_actions = trainer.actor_forward_for_sarsa(batch)
                
                advantages = evaluator.evaluate_sarsa(batch, mortality, rl_actions)
            
            # # check that advantages['survival_n'] == batch_losses['survival_n'] and advantages['non_survival_n'] == batch_losses['non_survival_n'], "Mismatch in survival/non-survival counts between batch_losses and advantage_log_dict"
            # if not (advantages['survival_n'] == batch_losses['survival_n'] and advantages['non_survival_n'] == batch_losses['non_survival_n']):
            #     logger.warning(f"Mismatch in survival/non-survival counts between batch_losses and advantages for batch_idx {batch_idx}. batch_losses: {batch_losses}, advantages: {advantages}")

        else:
            advantages = None        
                
        # Create pairs of (incoming_batch_data, target_epoch_container)
        if q_stats is not None:
            metric_pairs = [
                (batch_losses, val_losses),
                (q_stats, val_q_stats),
                (neuron_stats, val_neuron_stats),
            ]
        else:
            metric_pairs = [
                (batch_losses, val_losses),
                (neuron_stats, val_neuron_stats),
            ]

        # Dynamically add advantages to the processing list if it exists
        if advantages is not None:
            metric_pairs.append((advantages, val_advantages))
                
        # Clean, single loop to add the prefix and append values
        for batch_dict, epoch_dict in metric_pairs:
            for key, value in batch_dict.items():
                new_key = f"val/{key}"
                
                # If your container uses a standard dict, initialize the list if it's new
                if new_key not in epoch_dict:
                    epoch_dict[new_key] = []
                    
                epoch_dict[new_key].append(value)

    # A list of your actual dictionary variables to process in place
    if q_stats is not None:
        val_containers = [val_losses, val_q_stats, val_neuron_stats]
    else:
        val_containers = [val_losses, val_neuron_stats]        
    
    if advantages is not None:
        val_containers.append(val_advantages)

    for epoch_dict in val_containers:
        # 1. Overwrite the lists of batch values with their single calculated mean
        for key, list_of_values in epoch_dict.items():
            epoch_dict[key] = float(np.mean([
                v.item() if hasattr(v, "item") else v 
                for v in list_of_values
            ]))
            
        # 2. Add the epoch marker directly to this specific dictionary
        epoch_dict["epoch"] = epoch
        
        # 3. Log it immediately to W&B
        if wandb.run is not None:
            wandb.log(epoch_dict)
    
    return val_losses, val_q_stats, val_neuron_stats, val_advantages

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
    if config_manager.model_type in ['basic_q', 'fqe']:  # FQE's trainer is also a Q_Learner -- checkpoints its critic, not an actor
        return trainer.critic.state_dict().copy()
    elif config_manager.model_type in ['iql', 'iql_with_sarsa']:
        return {
            'critic': trainer.critic.state_dict().copy(),
            'v_net': trainer.v_net.state_dict().copy()
        }
    elif config_manager.model_type in ['bc_with_logprobs']:
        return {
            'actor': trainer.actor.state_dict().copy()
        }
    else:
        return trainer.actor.state_dict().copy()
    
def restore_model_parameters(trainer, best_model_state, config_manager):
    if config_manager.model_type in ['basic_q', 'fqe']:  # see copy_model_parameters
        trainer.critic.load_state_dict(best_model_state)
    elif config_manager.model_type in ['iql', 'iql_with_sarsa']:
        trainer.critic.load_state_dict(best_model_state['critic'])
        trainer.v_net.load_state_dict(best_model_state['v_net'])
    elif config_manager.model_type in ['bc_with_logprobs']:
        trainer.actor.load_state_dict(best_model_state['actor'])
    else:
        trainer.actor.load_state_dict(best_model_state) # for td3-bc I imagine we need both models!!!!!! SO FIX!!!!!!
    

def single_split_training(config_manager): 
    """ 
    based on supervised training.py file - at some point we might want to merge these two files and have a flag for whether to do RL or supervised training, but for now keeping separate for clarity.
    
    We need to select what input (state) data to use here - either the windowed data from the ShardProcessor,
    or the transformer output from dynamic deephit. We can specify this in the config_manager and then create the appropriate processor here.
    
    """
    # check environment - if local, we use wandb in online mode, if cluster, we use wandb in offline mode and then sync after training
    if config_manager.environment == "local":
        config_manager.training_config['wandb_mode'] = "online"
    else:
        config_manager.training_config['wandb_mode'] = "online" if can_reach_wandb() else "offline"
        logger.info(f"HPC wandb connectivity check: mode set to '{config_manager.training_config['wandb_mode']}'")
    
    logger.info("Starting single split training...")
    if config_manager.training_config.get('discount') is not None:
        logger.info("Discount set to: " + str(config_manager.training_config['discount']))
        
    # create processors
    # NEED TO ADD CONFIGURATION OPTION TO USE THE FEATURE DATA OR THE TRANSFORMER OUTPUT FROM DYNAMIC DEEPHIT MODEL
    # NEED TO ADD REWARD PROCESSING HERE
    ################ MOVED ###############################
    # train_processor, val_processor, test_processor = create_processor_from_shards(
    #     config_manager, 
    #     normalize=True
    #     )

    early_stopping = get_early_stopping_config(config_manager)
    if config_manager.search_variant == 'random_search':
        logger.info(" Hyperparameter tuning mode enabled.")

        train_processor, val_processor, test_processor = create_processor_from_shards(
                        config_manager, 
                        normalize=True
                    )

        # hyperparameter optimization
        best_val_loss = optimize_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager
        )

        if best_val_loss is not None:
             # save best_val_loss to a csv file in the experiment directory
            best_val_loss_path = config_manager.experiment_dir / "best_val_loss.csv"
            best_val_loss.to_csv(best_val_loss_path, index=False)
            logger.info(f"Best validation losses saved to {best_val_loss_path}")
        else:
            logger.warning("No valid validation losses were recorded during hyperparameter optimization.")

    elif config_manager.search_variant == 'single_hyperparameter_sweep':
        logger.info(" Single hyperparameter sweep mode enabled.")

        for value in config_manager.values:
            hyperparams = get_fixed_hyperparameters(config_manager)
            hyperparams[config_manager.hyperparameter_to_sweep] = value
            logger.info(f"Training with {config_manager.hyperparameter_to_sweep}={value}...")

            train_processor, val_processor, test_processor = create_processor_from_shards(
                            config_manager, 
                            normalize=True
                        )

            # train model
            train_with_hyperparameters(
                train_processor, 
                val_processor, 
                config_manager, 
                hyperparams,
            )        

    elif config_manager.search_variant == 'best_hyperparameters' and \
        config_manager.seeds is not None and len(config_manager.seeds) > 0:
        logger.info(" Best hyperparameter training mode enabled.")

        # train model with best hyperparameters for each seed
        for seed in config_manager.seeds:
            config_manager.training_config['seed'] = seed
            logger.info(f"Training with best hyperparameters for seed {seed}...")

            # get fixed hyperparameters
            hyperparams = get_fixed_hyperparameters(config_manager)

            train_processor, val_processor, test_processor = create_processor_from_shards(
                config_manager, 
                normalize=True
            )

            # train model
            train_with_hyperparameters(
                train_processor, 
                val_processor, 
                config_manager, 
                hyperparams,
            )       

    elif config_manager.search_variant == 'best_hyperparameters_fixed_cql_alpha' and \
        config_manager.seeds is not None and len(config_manager.seeds) > 0:
        logger.info(" Best hyperparameter training mode enabled.")

        # train model with best hyperparameters for each seed
        for seed in config_manager.seeds:
            config_manager.training_config['seed'] = seed
            logger.info(f"Training with best hyperparameters for seed {seed}...")

            # get fixed hyperparameters
            hyperparams = get_fixed_hyperparameters(config_manager)
            hyperparams['cql_alpha'] = config_manager.cql_alpha  # override cql_alpha with fixed value
            config_manager.training_config['cql_alpha'] = config_manager.cql_alpha  # also set in training_config for logging

            train_processor, val_processor, test_processor = create_processor_from_shards(
                            config_manager, 
                            normalize=True
                        )

            # train model
            train_with_hyperparameters(
                train_processor, 
                val_processor, 
                config_manager, 
                hyperparams,
            )                 

    else:
        logger.info(" Fixed hyperparameter training mode enabled.")

        # get fixed hyperparameters
        hyperparams = get_fixed_hyperparameters(config_manager)

        train_processor, val_processor, test_processor = create_processor_from_shards(
                        config_manager, 
                        normalize=True
                    )

        # train model
        train_with_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager, 
            hyperparams,
        )


def optimize_hyperparameters(train_processor, val_processor, config_manager):
    """for hyperparameter optimization."""
    
    early_stopping = get_early_stopping_config(config_manager)
    if early_stopping['flag']:
        # create dictionary for best vall
        best_loss = []

    # create hyperparameter space
    hyperparameter_space = get_hyperparameter_space(config_manager)

    # Keep only best trajectories if a fraction to keep is specified
    if config_manager.model_type == "any_percent_bc":
        logger.info(f"    Selecting best trajectories (frac={config_manager.frac})")
        select_best_trajectories(train_processor, config_manager)
        select_best_trajectories(val_processor, config_manager)    
    
    ##################### loop through hyperparameter combinations #####################
    # dedicated RNGs for hyperparameter sampling only — created once, outside
    # the loop, so set_seed() calls inside training (per-iteration) can never
    # reset them and cause repeated draws.
    search_seed = int.from_bytes(os.urandom(4), "big")  # genuinely random, not time-based
    logger.info(f"Hyperparameter search seed: {search_seed}")
    config_manager.training_config['search_seed'] = search_seed

    sampler_rng = random.Random(search_seed)
    sampler_np_rng = np.random.default_rng(search_seed)
    logger.info(f"Starting hyperparameter optimization with {config_manager.n_search_iterations} iterations...")
    
    for i in range(config_manager.n_search_iterations):
        ################# select hyperparameters randomly #################
        current_params = select_hyperparameters_randomly(
            hyperparameter_space, sampler_rng, sampler_np_rng
        )       
        logger.info(f"Testing hyperparameters {i+1}/{config_manager.n_search_iterations}: {current_params}")

        ################# Train with current hyperparameters #################      
        if early_stopping['flag']:
            all_scores, model_dir, best_val_loss = train_with_hyperparameters(
                train_processor, 
                val_processor, 
                config_manager, 
                current_params,
                search_iteration=i
            )
            best_val_loss['model_dir'] = str(model_dir) 
            best_loss.append(best_val_loss)
        
        else:
            all_scores, model_dir = train_with_hyperparameters(
                train_processor, 
                val_processor, 
                config_manager, 
                current_params,
                search_iteration=i
            )
    
    if early_stopping['flag']:
        if len(best_loss) > 0:
            # val_loss = {'loss_type': config_manager.training_config['loss_metric']}
            # for loss_dict in best_loss:
            #     val_loss[loss_dict['model_dir']] = loss_dict[val_loss['loss_type']]
            # create a dataframe from best_loss and sort by the loss_type, with columns ordered model_dir, config_manager.training_config['loss_metric']
            best_loss_df = pd.DataFrame(best_loss)
            best_loss_df = best_loss_df.sort_values(by=config_manager.training_config['loss_metric'])
            best_loss_df = best_loss_df[['model_dir', config_manager.training_config['loss_metric']]]

            return best_loss_df
        else:
            return None   
    else:
        return None
        

def has_reached_minimum(loss_history, loss_type="critic_loss", split="val", patience=5, min_delta=0.0):
    """
    Returns True once loss_history has gone `patience` epochs without
    achieving a new minimum better than min_delta improvement.
    This is standard early-stopping logic, repurposed as a
    'value function has converged' signal rather than a stop-training signal.
    """
    key = f"{split}/{loss_type}"
    if len(loss_history[key]) < patience + 1:
        return False
    best_so_far = min(loss_history[key][:-patience])
    recent_best = min(loss_history[key][-patience:])
    return recent_best >= best_so_far - min_delta


def is_stably_converged(loss_history, loss_type="v_loss", split="val", burn_in=10, window=10, rel_tol=0.05):
    """
    Returns True once loss_history has been relatively flat (std/mean < rel_tol)
    over the last `window` epochs, but only evaluated AFTER `burn_in` epochs
    have passed — avoiding the early near-zero/rising phase being mistaken
    for convergence.
    """
    key = f"{split}/{loss_type}"
    if len(loss_history[key]) < burn_in + window:
        return False
    
    recent = np.array(loss_history[key][-window:])
    rel_variability = recent.std() / (abs(recent.mean()) + 1e-8)
    return rel_variability < rel_tol


def build_zero_occupancy_action_pool(actions, min_count=1):
    """
    Identifies action-space bins with near-zero occupancy in the dataset,
    and returns their bounds so we can sample actions uniformly within
    each unsupported bin. joint_counts: (n_fluid_bins, n_vaso_bins) array.

    Currently hardcoded to 2D action space (fluid, vaso), bins each into 
    20 bins, so joint_counts is a 20x20 array. The function returns a list of tuples,

    """

    n_bins = 20
    joint_counts, fluid_edges, vaso_edges = np.histogram2d(actions[:, 0], actions[:, 1], bins=n_bins)   

    zero_bins = [] # list of (f_lo, f_hi, v_lo, v_hi)
    for i in range(joint_counts.shape[0]):
        for j in range(joint_counts.shape[1]):
            if joint_counts[i, j] < min_count:
                zero_bins.append((
                    fluid_edges[i], fluid_edges[i + 1],
                    vaso_edges[j], vaso_edges[j + 1],
                ))

    return zero_bins


def train_with_hyperparameters(
        train_processor, 
        val_processor, 
        config_manager, 
        hyperparams, 
        search_iteration=None,
    ):
    """Train model with specific hyperparameters and return validation score."""
    
    # process the reward
    health_score_median = train_processor.process_reward(config_manager, hyperparams)
    val_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
    # add reward_type to hyperparams for logging
    hyperparams['reward_type'] = config_manager.reward_type
    
    n_epochs = config_manager.n_epochs

    batch_size = hyperparams['batch_size']
    loss_metric = config_manager.training_config['loss_metric']   

    # get job_id and seed
    job_id = get_job_id(config_manager)
    # check if config_manager.training_conifg['seed'] is set
    if config_manager.training_config.get('seed') is not None:
        seed =  config_manager.training_config['seed']
    else:
        if hyperparams.get('use_fixed_seed', False):
            seed = hyperparams['seed']  # Default to 42 if not specified
        else:
            seed = generate_run_seed(job_id, search_iteration)
        config_manager.seed = seed

    hyperparams['seed'] = seed
    set_seed(seed)
   
    # initialize wandb for this training run
    run = wandb_init(hyperparams, job_id)

    hyperparams['wandb_id'] = run.id if run is not None else f"{job_id}_no_wandb"
    print(f"W&B run initialized with ID: {hyperparams['wandb_id']}")

    # Create replay buffers
    training_buffer, scaler = create_buffer(
        train_processor, 
        batch_size, 
        hyperparams,
        action_names=config_manager.data_config['actions'],
    )
    # add ratio_of_fluid_to_vaso to config_manager for actor loss
    actions = train_processor.rl_dataset["actions"]

    # get zero_occupancy_action_pool
    zero_bins = build_zero_occupancy_action_pool(actions, min_count=1)
    zero_bins_tensor = torch.tensor(zero_bins, dtype=torch.float32)

    ratio_of_fluid_to_vaso = (torch.sum(actions[:, 0]) / (torch.sum(actions[:, 1]) + 1e-8)).item()  # avoid division by zero
    config_manager.training_config['ratio_of_fluid_to_vaso'] = ratio_of_fluid_to_vaso

    val_buffer, _ = create_buffer(
        val_processor,
        batch_size, 
        hyperparams,
        scaler=scaler, 
        action_names=config_manager.data_config['actions'],
    )
    
    # get total number of steps for scheduler if needed
    hyperparams['total_steps'] = n_epochs * len(training_buffer)
    
    if config_manager.model_type == "fqe":
        config_manager.actor_type = "cql_with_logprobs"  # 
        actor = load_actor(train_processor, hyperparams, config_manager)

        trainer = create_trainer(train_processor, hyperparams, config_manager, actor=actor)
    else:    
        # create trainer
        trainer = create_trainer(train_processor, hyperparams, config_manager)
        trainer.zero_occupancy_action_pool = zero_bins_tensor.to(trainer.device)  # Assign the zero occupancy action pool to the trainer

    # create sarsa_evaluator if model_type is not basic_q, since we want to evaluate the advantage of the learned policy compared to the clinician policy during training
    if config_manager.model_type != "basic_q" and config_manager.model_type != "fqe":
        evaluator = create_evaluator(config_manager, train_processor)
    else:
        evaluator = None
    
        
    # Early stopping configuration 
    early_stopping = get_early_stopping_config(config_manager)

    best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training with early stopping (patience={early_stopping['patience']}, eval_freq={early_stopping['eval_freq']}, min_delta={early_stopping['min_delta']})")

    # create TrainingMetrics instance to store all training and validation metrics in a structured way for easy access and saving at the end of training
    training_metrics = TrainingMetrics()

    """ evaluate before any training for epoch 0 - the generated values are calculated 
    per batch and concatenated onto the defaultdict lists """
    val_losses, val_q_stats, val_neuron_stats, val_advantages = evaluate_during_training(
        trainer, 
        evaluator, 
        -1, 
        val_buffer, 
        batch_size
    )

    training_metrics.append_epoch_metrics(
        val_losses=val_losses,
        val_q_stats=val_q_stats,
        val_advantages=val_advantages,
        val_neuron_stats=val_neuron_stats        
    )

    current_val_loss = np.mean(training_metrics.get_current_epoch_val_loss(loss_metric))
    logger.info(f"Epoch {-1}: Val {loss_metric}={current_val_loss:.4f}")
    
    # # optionally, estimate V0 and append to validation losses
    # if config_manager.training_config['monitor_v0']:
    #     estimate_v0(trainer, val_starting_states, batch_size, val_losses, prefix='val')
    
    ############################### Training loop #################################
    testing = getattr(config_manager, 'testing', False)
    if testing:
        n_epochs_testing = 100
        logger.info(f" Testing mode enabled: will limit training to {n_epochs_testing} epochs and 100 batches per epoch.")
        n_epochs = min(n_epochs, n_epochs_testing)
    
    torch.cuda.empty_cache()

    global_step = 0
    for epoch in range(n_epochs):
        # print epoch
        logger.info(f"Epoch {epoch+1}/{n_epochs}")  
        
        epoch_training_losses = defaultdict(list)
        epoch_training_q_stats = defaultdict(list)
        epoch_grad_norm = defaultdict(list)

        for batch_idx, batch in enumerate(training_buffer):
            batch_losses, q_stats, grad_norm = trainer.train(batch)            

            # 1. Prepare a lightweight dictionary for W&B for THIS specific iteration
            batch_payload = {
                "epoch": epoch,  # Keep your tracking matched to your logger display
                "iteration": global_step,
            }

            # 2. Process and map your dictionaries into the payload and your aggregators
            # Pairing (incoming_batch_dict, target_epoch_aggregator)
            if q_stats is not None:
                metric_pairs = [
                    (batch_losses, epoch_training_losses),
                    (q_stats, epoch_training_q_stats),
                    (grad_norm, epoch_grad_norm),
                ]
            else:
                metric_pairs = [
                    (batch_losses, epoch_training_losses),
                    (grad_norm, epoch_grad_norm),
                ]

            for batch_dict, epoch_container in metric_pairs:
                for key, value in batch_dict.items():
                    # Extract raw numerical value if it is a PyTorch tensor
                    scalar_val = value.item() if hasattr(value, "item") else value

                    # Apply the specific prefix we defined in wandb_init
                    new_key = f"train/batch_{key}"

                    # Store the scalar in the payload for W&B
                    batch_payload[new_key] = scalar_val

                    # Append to your aggregator so we can compute epoch means later
                    epoch_container[new_key].append(scalar_val)
            
            # 3. Stream the single-batch iteration metrics directly to W&B
            if wandb.run is not None:
                wandb.log(batch_payload)        

            global_step += 1

            # check if we are simply testing the training loop, in which case we break after 100 batches
            if batch_idx >= 10 and testing:
                logger.info(" Testing mode: breaking after 10 batches.")
                break

        # log the batch values to the TrainingMetrics object for later epoch-level processing
        training_metrics.append_epoch_metrics(
            batch_training_losses=epoch_training_losses,
            batch_training_q_stats=epoch_training_q_stats,
            batch_training_grad_norms=epoch_grad_norm  
        )    

        # calculate epoch averages and log to W&B
        # 1. Gather your training containers to process them in place
        train_epoch_containers = [
            epoch_training_losses, 
            epoch_training_q_stats, 
            epoch_grad_norm
        ]

        for epoch_dict in train_epoch_containers:
            # 2. Overwrite the lists of batch values with their single calculated mean
            # We use list(epoch_dict.keys()) so we can safely modify the dict while looping
            for batch_key in list(epoch_dict.keys()):
                list_of_values = epoch_dict[batch_key]
                
                # Compute the average
                mean_value = float(np.mean(list_of_values)) if list_of_values else 0.0
                
                # Transform the key name from 'train/batch_loss' to 'train/epoch_loss'
                epoch_key = batch_key.replace("train/batch_", "train/epoch_")
                
                # Store the average under the new key, and clean up the old batch-list key
                epoch_dict[epoch_key] = mean_value
                del epoch_dict[batch_key]
        
            # 3. Add the structural tracking tokens to the dictionary
            epoch_dict["epoch"] = epoch       # Will be 0 on your first pass
            epoch_dict["global_step"] = global_step - 1
        
            # 4. Send the epoch summary point to W&B
            if wandb.run is not None:
                wandb.log(epoch_dict)
        
        # log the epoch values to the TrainingMetrics instance
        training_metrics.append_epoch_metrics(
            epoch_training_losses=epoch_training_losses,
            epoch_training_q_stats=epoch_training_q_stats,
            epoch_training_grad_norms=epoch_grad_norm  
        )
        
        epoch_loss = training_metrics.get_current_epoch_train_loss(loss_metric)
        logger.info(f"Epoch {epoch}: Train {loss_metric}={epoch_loss:.4f}")
        epoch_grad_norm = training_metrics.get_current_epoch_grad_norm()
        if epoch_grad_norm is not None:
            logger.info(f"Epoch {epoch}: Gradient norms: {epoch_grad_norm}")
                
        #################### Evaluate ####################
        # Evaluate periodically
        if epoch % config_manager.training_config['eval_freq'] == 0 \
                                                or epoch == n_epochs - 1:
            
            val_losses, val_q_stats, val_neuron_stats, val_advantages = evaluate_during_training(
                trainer, 
                evaluator, 
                epoch, 
                val_buffer, 
                batch_size
            )

            training_metrics.append_epoch_metrics(
                val_losses=val_losses,
                val_q_stats=val_q_stats,
                val_advantages=val_advantages,
                val_neuron_stats=val_neuron_stats        
            )
            
            current_val_loss = training_metrics.get_current_epoch_val_loss(loss_metric)
            logger.info(f"Epoch {epoch}: Val {loss_metric}={current_val_loss:.4f}")
                                 
            # if model_type is iql, we need to check for whether v_net and critic loss have
            # converged, so we can start training the actor
            if config_manager.model_type in ["iql", "iql_with_sarsa"]:
                if not trainer.train_actor_flag:                   
                    if config_manager.model_type == "iql":
                        min_flag_value = is_stably_converged(training_metrics.val_losses, loss_type="v_loss", burn_in=10, window=10, rel_tol=0.05)
                        min_flag_critic = has_reached_minimum(training_metrics.val_losses, loss_type="critic_loss", patience=10, min_delta=0.0)
                
                        if min_flag_value and min_flag_critic:
                            trainer.set_actor_active(True)
                            logger.info(f"Epoch {epoch}: V and Critic losses have converged, starting to train actor.")
                    
                    else: # iql_with_sarsa
                        min_flag_value = has_reached_minimum(training_metrics.val_losses, loss_type="v_loss", patience=10, min_delta=0.0)
                        if min_flag_value:
                            trainer.set_actor_active(True)
                            logger.info(f"Epoch {epoch}: V loss has converged, starting to train actor.")


            # Check for improvement and copy model parameters if best, only if early stopping is enabled
            if early_stopping['flag']:
                improvement, best_val_loss = check_improvement(best_val_loss, current_val_loss, early_stopping)

                if improvement > 0:
                    best_model_state = copy_model_parameters(trainer, config_manager)
                    
                #  Early stopping - stop training THIS hyperparameter combination
                if early_stopping['patience_counter'] >= early_stopping['patience']:                
                    logger.info(f" Early stopping at epoch {epoch}/{n_epochs}")
                    logger.info(f" Best validation loss: {best_val_loss:.4f}")

                    n_epochs = epoch + 1  # Update n_epochs to actual number run

                    break
        
    ######################### training complete ########################
    # restore best model if using early stopping. 
    if early_stopping['flag']:
        if best_model_state is not None:
            restore_model_parameters(trainer, best_model_state, config_manager)
            logger.info(f"    Restored best model (val_loss={best_val_loss:.4f})")   

            ######### ADD BEST VAL_LOSS TO TRAINING METRICS #########
            training_metrics.best_val_loss = {loss_metric: best_val_loss}
    
    # model dir
    model_dir = config_manager.experiment_dir / hyperparams['wandb_id']
    # create model dir if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)    

    # run full evaluation on validation set
    final_losses, final_q_stats, final_neuron_stats, final_val_advantages = evaluate_during_training(
        trainer, 
        evaluator, 
        epoch, 
        val_buffer, 
        batch_size,
        final=True
    )

    training_metrics.append_epoch_metrics(
        final_val_losses=final_losses,
        final_val_q_stats=final_q_stats,
        final_val_advantages=final_val_advantages,
        final_val_neuron_stats=final_neuron_stats        
    )
    
    # print key value pairs from training_metrics.final_val_losses to logger
    logger.info("Final val metrics:")
    for key, value in training_metrics.final_val_losses.items():
        logger.info(f" {key}: {value[-1]:.4f}")

    model_path = Path(model_dir / "final_model.pt")
    
    if config_manager.model_type in ['basic_q', 'fqe']:  # FQE's trainer is also a Q_Learner -- same checkpoint shape as SARSA
        checkpoint = {
            'total_it': trainer.total_it,
            'critic_state_dict': trainer.critic.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'training_config': config_manager.training_config,
        }
        torch.save(checkpoint, model_path)

    elif config_manager.model_type in ['bc_with_logprobs']:
        checkpoint = {
            'total_it': trainer.total_it,
            'actor_state_dict': trainer.actor.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'training_config': config_manager.training_config,
        }
        torch.save(checkpoint, model_path)
    
    elif config_manager.model_type in ['td3_bc', 'conservative_q_learning', 
                                       'cql_with_logprobs']:
        checkpoint = {
            'total_it': trainer.total_it,
            'actor_state_dict': trainer.actor.state_dict(),
            'critic_state_dict': trainer.critic.state_dict(),
            'actor_target_state_dict': trainer.actor_target.state_dict(),
            'critic_target_state_dict': trainer.critic_target.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'training_config': config_manager.training_config,
        }
        torch.save(checkpoint, model_path)

    elif config_manager.model_type in ['iql', 'iql_with_sarsa']:
        if hasattr(trainer, 'critic_optimizer'):
                checkpoint = {
                'total_it': trainer.total_it,
                'actor_state_dict': trainer.actor.state_dict(),
                'critic_state_dict': trainer.critic.state_dict(),
                'v_net_state_dict': trainer.value_net.state_dict(),
                'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
                'v_net_optimizer_state_dict': trainer.value_net_optimizer.state_dict(),
                'hyperparameters': hyperparams,
                'training_config': config_manager.training_config,
            }
            
        else:        
            checkpoint = {
                'total_it': trainer.total_it,
                'actor_state_dict': trainer.actor.state_dict(),
                'critic_state_dict': trainer.critic.state_dict(),
                'v_net_state_dict': trainer.value_net.state_dict(),
                'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
                'v_net_optimizer_state_dict': trainer.value_net_optimizer.state_dict(),
                'hyperparameters': hyperparams,
                'training_config': config_manager.training_config,
            }
        torch.save(checkpoint, model_path)
    
    elif config_manager.model_type == 'cvp':
        checkpoint = {
            'total_it': trainer.total_it,
            'actor_state_dict': trainer.actor.state_dict(),
            'critic_state_dict': trainer.critic.state_dict(),
            'critic_target_state_dict': trainer.target_critic.state_dict(),
            'actor_optimizer_state_dict': trainer.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': trainer.critic_optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'training_config': config_manager.training_config,
        }
        torch.save(checkpoint, model_path)
    
    else:
        torch.save(trainer.model.state_dict(), model_path)
    logger.info(f" Final model saved to {model_path}")

    # plot losses
    plot_results(training_metrics, model_dir, config_manager)

    # save results
    all_scores = save_results(
        training_metrics,
        hyperparams,
        n_epochs,
        model_dir,
        config_manager
    )

    if early_stopping['flag']:
        logger.info(f" Best validation loss: {best_val_loss:.4f}")
        best_val_loss = training_metrics.best_val_loss

        return all_scores, model_dir, best_val_loss
        
    else:
        return all_scores, model_dir