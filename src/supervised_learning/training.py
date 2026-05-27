
import wandb
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import pickle
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
from ..utils.logging_utilities import wandb_init, generate_run_seed, get_job_id
from ..data.data_processor import DataProcessor
from ..data.data_processor_for_shards import ShardProcessor
from ..rl.replay_buffer import DynamicDeepHit_Lazy, worker_init_fn, DynamicDeepHit_InMemory, DynamicDeepHitTransformerDataset

from ..utils.utilities import set_seed
from ..rl.plotting import plot_c_and_brier_over_time, plot_auroc_prc_curves, plot_results
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
    "dynamic_deephit_transformer": "src.supervised_learning.models.dynamic_deephit_transformer",
    "bc_transformer": "src.supervised_learning.models.bc_transformer"
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


class TrainingMetrics:
    def __init__(self):
        self._metric_keys = [
            "batch_training_losses",
            "epoch_training_losses",
            "val_losses",
            "final_val_losses",
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
                not incoming_dict or len(list(incoming_dict.values())[0]) == 0
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
            # selected_params[param] = round(value.rvs(), 10)  # round for readability
            selected_params[param] = value.rvs()  # round for readability
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


def create_processor_from_shards(config_manager, normalize=True, set_to_return='all'):
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
    

def create_buffer(processor, batch_size, config_manager, shuffle=True):
            
    dataset = DynamicDeepHitTransformerDataset(processor, config_manager)
    buffer = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        # num_workers=0,          # reduce to save memory
        # prefetch_factor=None,      # small prefetch to limit RAM
        # pin_memory=False,       # set True if transfers dominate
        )

    return buffer, dataset


def create_trainer(train_processor, hyperparams, config_manager):

    # feature_dim = train_processor.dynamic_deephit_dataset["features"].shape[2]
    if config_manager.model_type == 'dynamic_deephit_transformer':
        feature_dim = train_processor.dynamic_deephit_dataset['feature_dim'] 
    else:
        feature_dim = train_processor.dynamic_deephit_dataset['feature_dim'] - len(config_manager.training_config['actions'])
    if config_manager.model_type == 'bc_transformer':
        num_targets = len(hyperparams['actions']) # hardcoded for now
    else:
        num_targets = len(config_manager.training_config['actions']) # hardcode for now

    model_type = config_manager.model_type
    module = importlib.import_module(MODEL_MODULES[model_type])
    
    return module.get_trainer(
        config_manager,
        feature_dim,
        hyperparams,
        num_targets=num_targets
    )


def append_epoch_losses(
        epoch_training_losses, 
        training_losses
    ):
    """Calculates average losses for the epoch and appends to training_losses."""      
    epoch_avg_losses = {key: np.mean(values) for key, values in epoch_training_losses.items()}
    # Append epoch averages to training_losses
    for key, value in epoch_avg_losses.items():
        training_losses[key].append(value)     



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

    val_scores = {**training_metrics.final_val_losses, **training_metrics.final_val_neuron_stats}
    
    # create a dictionary to save the scores
    all_scores = {
        "val_scores": val_scores,
        "training_scores": training_losses,
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
            'training_scores': training_losses,
            'config': config_manager.training_config
        }), f, indent=2)
    
    logger.info(f"✓ Training complete. Results saved to {results_path}")

    return all_scores



def evaluate_model(
    trainer, 
    dataloader, 
    config_manager,
    max_samples=None,
    return_full_curves=False,
    compute_survival_metrics_flag=True,
    epoch=None
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

    Note that mask logic is reversed here as we are no longer using a mask for the
    transformer (inputs now padded with data), only for loss, so True indicates valid data.
        
    """
    eval_losses = defaultdict(list)
    all_logits = []
    all_time_to_event = []
    all_event_labels = []
    
    num_samples = 0
    
    testing = getattr(config_manager, 'testing', False)
    for batch_idx, batch in enumerate(dataloader):
        # Check if we've reached max samples
        if max_samples is not None and num_samples >= max_samples:
            break
        
        batch_losses, logits = trainer.evaluate(batch, epoch=epoch)
        
        # Collect batch losses
        for key, value in batch_losses.items():
            eval_losses[key].append(value)
        
        
        time_to_event = batch['time_to_event']
        mask = batch['loss_mask']
        if config_manager.model_type == 'dynamic_deephit_transformer' and \
            config_manager.training_config.get('final_timestep_only', True):
            time_to_event = time_to_event[:, -1]
            mask = mask[:, -1]


        event_labels = batch['event']
        
        # if logits has 4 dimension (i.e. transformer version), reshape to 3D by merging first two dimensions
        if logits is not None and logits.dim() == 4:
            B, T_cap, R, D = logits.shape
            logits = logits.reshape(B * T_cap, R, D)
            # also reshape time_to_event and event_labels accordingly
            time_to_event = time_to_event.reshape(B * T_cap)
            
            if event_labels.dim() == 1:
                event_labels = event_labels.unsqueeze(1).expand(-1, T_cap)  # [B, T_cap]
            event_labels = event_labels.reshape(B * T_cap)

            # get mask and reshape
            mask = mask.reshape(B * T_cap)
            logits = logits[mask]
            time_to_event = time_to_event[mask]
            event_labels = event_labels[mask]

            num_samples = len(event_labels)
        else:
            # GRU version, we need to eliminate all but the last time_to_event value
            time_to_event = time_to_event[:, -1]
            num_samples += batch['features'].size(0)

        # Collect data for survival metrics
        if logits is not None and compute_survival_metrics_flag:
            all_logits.append(logits.cpu())
            # all_time_to_event.append(batch[1].cpu())
            all_time_to_event.append(time_to_event.cpu())
            # all_event_labels.append(batch[2].cpu())
            all_event_labels.append(event_labels.cpu())

        if testing and batch_idx >= 10:
            break  # Limit to 2 batches in testing mode

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


def evaluate_during_training(
        trainer, 
        epoch, 
        val_buffer, 
        batch_size,
        final=False
    ):
    """Evaluate model during training and log results."""

    val_losses = defaultdict(list)
    val_neuron_stats = defaultdict(list)
    
    for batch_idx, batch in enumerate(val_buffer):      
        if not final:
            if batch_idx * batch_size >= 50000:  # Limit validation to 5000 samples for speed, make this a variable
                break
       
        batch_losses, _, neuron_stats = trainer.evaluate(batch)

        # Create pairs of (incoming_batch_data, target_epoch_container)
        metric_pairs = [
            (batch_losses, val_losses),
            (neuron_stats, val_neuron_stats),
        ]

        # Clean, single loop to add the prefix and append values
        for batch_dict, epoch_dict in metric_pairs:
            for key, value in batch_dict.items():
                new_key = f"val/{key}"
                
                # If your container uses a standard dict, initialize the list if it's new
                if new_key not in epoch_dict:
                    epoch_dict[new_key] = []
                    
                epoch_dict[new_key].append(value)
            
        # A list of your actual dictionary variables to process in place
        val_containers = [val_losses, val_neuron_stats]

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
        wandb.log(epoch_dict)
    
    return val_losses, val_neuron_stats
    
    
    
    
    
    


# def evaluate_final_model(trainer, test_buffer, config_manager):
#     """Evaluate final model on test set."""
    
#     # Evaluate on full test set (no sample limit)
#     test_metrics = evaluate_model(
#         trainer,
#         test_buffer,
#         config_manager,
#         max_samples=None,
#         compute_survival_metrics_flag=True, 
#         return_full_curves=True
#     )
    
#     logger.info(f"Final test metrics: {test_metrics}")
    
#     return test_metrics


def check_improvement(best_val_loss, current_val_loss, early_stopping, change_direction='decrease'):
    """Checks if there is an improvement in validation loss and updates early stopping counters."""
    if change_direction == 'increase':
        improvement = current_val_loss - best_val_loss
    else:
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
    # check environment - if local, we use wandb in online mode, if cluster, we use wandb in offline mode and then sync after training
    if config_manager.environment == "local":
        config_manager.training_config['wandb_mode'] = "online"
    else:
        config_manager.training_config['wandb_mode'] = "offline"
    
    logger.info("Starting single split training...")

    # Create processors
    train_processor, val_processor, test_processor = create_processor_from_shards(
        config_manager, 
        normalize=True
        )
    
    # if config_manager.training_config['tuning'] and config_manager.n_search_iterations > 0:
    if config_manager.search_variant == 'random_search':
        logger.info(" Hyperparameter tuning mode enabled.")

        # hyperparameter optimization
        optimize_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager
        )

    else:
        logger.info(" Fixed hyperparameter training mode enabled.")

        # get fixed hyperparameters
        hyperparams = get_fixed_hyperparameters(config_manager)

        # train final model
        train_with_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager, 
            hyperparams,
        )




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
    # if not any(isinstance(v, list) or (isinstance(v, str) and '(' in v and ')' in v) 
    #            for v in config_manager.training_config.values()):
    #     raise ValueError("No hyperparameters to optimize. Ensure that config_manager.training_config contains lists of values to search over or distribution strings.")

    return hyperparameter_space


def keep_positive_trajectories(dataprocessor, config_manager):
    """
    This is for the behavioural cloning using terminal rewards only. In
    this case, it's simply a matter of keeping only survival trajectories;
    non-survival trajectories are discarded.
    """
    max_traj_len = config_manager.training_config['max_traj_len']

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
                if len(cur_ids) <= max_traj_len:
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


def keep_best_trajectories(dataprocessor, config_manager):

    fraction_to_keep = config_manager.frac
    discount = config_manager.training_config['discount']
   
    # check that the rewards are not terminal only, i.e. only -1, 0, +1
    if set(np.unique(dataprocessor.rl_dataset["rewards"])).issubset({-1, 0, 1}):
        raise ValueError("Rewards appear to be terminal only (-1, 0, +1). Use keep_positive_trajectories instead.")
    
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    for i, (reward, done) in enumerate(zip(dataprocessor.rl_dataset["rewards"], 
                                           dataprocessor.rl_dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount     
        
        if done == 1.0:
            if len(cur_ids) <= config_manager.training_config['max_traj_len']:
                # don't use trajectories that are too long
                ids_by_trajectories.append(list(cur_ids))
                returns.append(cur_return)
            
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(fraction_to_keep * len(sort_ord)))]
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


def select_best_trajectories(processor, config_manager):
    """Keeps only the best trajectories based on config_manager.frac."""
    if config_manager.frac <= 0. or config_manager.frac > 1.:
        raise ValueError("config_manager.frac must be in (0, 1]")
    
    if config_manager.frac != 0.:
        keep_best_trajectories(processor, config_manager)
    else:
        keep_positive_trajectories(processor)


def optimize_hyperparameters(train_processor, val_processor, config_manager):
    """for hyperparameter optimization."""
    
    # create hyperparameter space
    hyperparameter_space = get_hyperparameter_space(config_manager)

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
        train_with_hyperparameters(
            train_processor, 
            val_processor, 
            config_manager, 
            current_params,
            search_iteration=i
        )
    
    



def get_hyperparameter(hyperparams, config_manager, param_name):
    """Get hyperparameter value from hyperparams or config_manager."""
    if param_name in hyperparams:
        return hyperparams[param_name]
    else:
        return config_manager.training_config[param_name]



def train_with_hyperparameters(
        train_processor, 
        val_processor, 
        config_manager, 
        hyperparams, 
        search_iteration=None
    ):
    """Train model with specific hyperparameters and return validation score."""
    
    # PROCESSING REWARDS NOT NECESSARY FOR SUPERVISED LEARNING
        
    n_epochs = config_manager.n_epochs

    batch_size = get_hyperparameter(hyperparams, config_manager, 'batch_size')
    loss_metric = config_manager.training_config['loss_metric']

    # get job_id and seed
    job_id = get_job_id(config_manager)
    seed = generate_run_seed(job_id, search_iteration)
    hyperparams['seed'] = seed
    config_manager.seed = seed
    set_seed(seed)

    # initialize wandb for this training run
    run = wandb_init(hyperparams, job_id)

    hyperparams['wandb_id'] = run.id if run is not None else None
    print(f"W&B run initialized with ID: {hyperparams['wandb_id']}")

    # Create replay buffers
    training_buffer, training_dataset = create_buffer(train_processor, batch_size, config_manager, shuffle=True)
    val_buffer, _ = create_buffer(val_processor, batch_size, config_manager, shuffle=False)
    
    # get total number of steps for scheduler if needed
    hyperparams['total_steps'] = n_epochs * len(training_buffer)
    
    # create trainer
    trainer = create_trainer(train_processor, hyperparams, config_manager)
        
    # Early stopping configuration 
    early_stopping = get_early_stopping_config(config_manager)

    if loss_metric in ['weighted_multi_C']:
        best_val_loss = 0.
    else:
        best_val_loss = float('inf')
    best_model_state = None

    logger.info(f"Training with early stopping (patience={early_stopping['patience']}, eval_freq={early_stopping['eval_freq']}, min_delta={early_stopping['min_delta']})")
    
    # Use defaultdict to automatically create new keys
    # training_losses = defaultdict(list) 
    # raw_grad_norms = defaultdict(list)

    # create TrainingMetrics instance to store all training and validation metrics in a structured way for easy access and saving at the end of training
    training_metrics = TrainingMetrics()


    # Evaluate before training starts to get initial validation loss
    # val_losses = defaultdict(list)
    # evaluate before any training for epoch 0
    val_losses, val_neuron_stats = evaluate_during_training(
        trainer, 
        -1, 
        val_buffer, 
        batch_size, 
        final=False
    )   

    training_metrics.append_epoch_metrics(
        val_losses=val_losses,
        val_neuron_stats=val_neuron_stats,
    )

    current_val_loss = np.mean(training_metrics.get_current_epoch_val_loss(loss_metric))
    logger.info(f"Epoch {-1}: Val {loss_metric}={current_val_loss:.4f}")
    
    ############################### Training loop #################################
    testing = getattr(config_manager, 'testing', False)
    if testing:
        logger.info(" Testing mode enabled: will limit training to 10 epochs and 100 batches per epoch.")
        n_epochs = min(n_epochs, 5)
    
    torch.cuda.empty_cache()
    
    global_step = 0
    for epoch in range(n_epochs):
        # print epoch
        logger.info(f"Epoch {epoch+1}/{n_epochs}")  

        # shuffle training dataset
        training_dataset.reshuffle_subsample()
        #################### Train ####################
        epoch_training_losses = defaultdict(list)
        epoch_grad_norm = defaultdict(list)

        for batch_idx, batch in enumerate(training_buffer):
            batch_losses, _, grad_norm = trainer.train(batch, epoch)
           
            # 1. Prepare a lightweight dictionary for W&B for THIS specific iteration
            batch_payload = {
                "epoch": epoch,  # Keep your tracking matched to your logger display
                "iteration": global_step,
            }

            # # Automatically append all metrics - creates new keys if needed
            # for key, value in batch_losses.items():
            #     epoch_training_losses[key].append(value)

            # epoch_raw_grad_norms['raw_grad_norm'].append(raw_grad_norm)

            # 2. Process and map your dictionaries into the payload and your aggregators
            # Pairing (incoming_batch_dict, target_epoch_aggregator)
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

            # current_lr = trainer.optimizer.param_groups[0]['lr']
            # if batch_idx % 10 == 0 and current_lr < hyperparams['learning_rate']:
            #     print(f"Step {batch_idx}: LR = {current_lr:.6f}\n")

            # check if we are simply testing the training loop, in which case we break after 100 batches
            if batch_idx >= 10 and testing:
                logger.info(" Testing mode: breaking after 10 batches.")
                break

        # # Calculate average loss for this epoch and append to training_losses
        # append_epoch_losses(epoch_training_losses, training_losses)
        # append_epoch_losses(epoch_raw_grad_norms, raw_grad_norms)

        # log the batch values to the TrainingMetrics object for later epoch-level processing
        training_metrics.append_epoch_metrics(
            batch_training_losses=epoch_training_losses,
            batch_training_grad_norms=epoch_grad_norm  
        )  

        # calculate epoch averages and log to W&B
        # 1. Gather your training containers to process them in place
        train_epoch_containers = [
            epoch_training_losses, 
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
            
            val_losses, val_neuron_stats = evaluate_during_training(
                trainer, 
                epoch, 
                val_buffer, 
                batch_size, 
                final=False
            )   

            training_metrics.append_epoch_metrics(
                val_losses=val_losses,
                val_neuron_stats=val_neuron_stats        
            )

            current_val_loss = training_metrics.get_current_epoch_val_loss(loss_metric)
            logger.info(f"Epoch {epoch}: Val {loss_metric}={current_val_loss:.4f}")
            # # Determine validation loss for early stopping
            # current_val_loss = val_losses[loss_metric][-1]            
            # logger.info(f"Epoch {epoch}: Train {loss_metric}={training_losses[loss_metric][-1]:.4f}, Val {loss_metric}={current_val_loss:.4f}")
                            
            # Check for improvement and copy model parameters if best, only if early stopping is enabled
            if early_stopping['flag']:
                if loss_metric in ['weighted_multi_C']:
                    # for c-index, higher is better
                    improvement, best_val_loss = check_improvement(best_val_loss, current_val_loss, early_stopping, change_direction='increase')
                else:
                    # for losses, lower is better
                    improvement, best_val_loss = check_improvement(best_val_loss, current_val_loss, early_stopping)

                    if improvement > 0:
                        best_model_state = copy_model_parameters(trainer, config_manager)
                        
                    #  Early stopping - stop training THIS hyperparameter combination
                    if early_stopping['patience_counter'] >= early_stopping['patience']:                
                        logger.info(f" Early stopping at epoch {epoch}/{n_epochs}")
                        logger.info(f" Best validation loss: {best_val_loss:.4f}")

                        n_epochs = epoch + 1  # Update n_epochs to actual number run

                        break
    
    ########################## training complete ########################################
    # restore best model if using early stopping. 
    if early_stopping['flag']:
        if best_model_state is not None:
            restore_model_parameters(trainer, best_model_state, config_manager)
            logger.info(f"    Restored best model (val_loss={best_val_loss:.4f})")   

    # model dir
    model_dir = config_manager.experiment_dir / hyperparams['wandb_id']
    # create model dir if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)   

    # run full evaluation on validation set
    final_losses, final_neuron_stats = evaluate_during_training(
        trainer, 
        epoch, 
        val_buffer, 
        batch_size,
        final=True
    )

    training_metrics.append_epoch_metrics(
        final_val_losses=final_losses,
        final_val_neuron_stats=final_neuron_stats        
    )

    # print key value pairs from training_metrics.final_val_losses to logger
    logger.info("Final val metrics:")
    for key, value in training_metrics.final_val_losses.items():
        logger.info(f" {key}: {value[-1]:.4f}")

    model_path = Path(model_dir / "final_model.pt")

    if config_manager.model_type in ['dynamic_deephit_transformer', 'bc_transformer']:
        checkpoint = {
            'total_it': trainer.total_it,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'hyperparameters': hyperparams,
            'training_config': config_manager.training_config,
        }
        torch.save(checkpoint, model_path)
    else:
        # no other models currently save optimizer state, but we can easily add this to the checkpoint in the future if needed
        raise NotImplementedError("Saving checkpoints with optimizer state is currently only implemented for transformer-based models. Add implementation for other model types if needed.")

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