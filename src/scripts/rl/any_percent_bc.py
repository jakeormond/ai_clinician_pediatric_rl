import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import os
import time


import pyrallis

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

from ...rl.training import outer_training_loop
from ...data.data_processor import DataProcessor


TensorBatch = List[torch.Tensor]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train mortality prediction model')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    
    parser.add_argument('--experiment_type', 
                        default='reinforcement_learning', 
                        choices=['supervised_learning', 'reinforcement_learning'], 
                        help='Type of experiment to run (default: reinforcement_learning)')
    
    parser.add_argument('--model_type', 
                        default='any_percent_bc', 
                        choices=['any_percent_bc'], 
                        help='Model type to train (default: any_percent_bc)') 

    parser.add_argument('--mode', type=str, default='single_fold',
                        choices=['single_fold', 'aggregate'],
                        help='Mode of training: single_fold or aggregate (default: single_fold)')
    
    # mode-specific arguments    
    parser.add_argument('--fold', type=int,
                       help='Fold number to train (required for single_fold mode)')

    parser.add_argument('--experiment_variant', 
                        default='quick_test', 
                        choices=['quick_test', 'full_test'], 
                        help='Variant of the experiment to run (default: quick_test)')

    parser.add_argument('--search_variant', 
                        default='random_search', 
                        choices=['random_search', 'grid_search'], 
                        help='Variant of the search to use (default: random_search)')

    parser.add_argument('--n_search_iterations', type=int, default=2,
                        help='Number of search iterations to run (default: 2)')
    
    parser.add_argument('--start_time', type=str, default = None,                  
                    help='Start time for this experiment run')
    
    args = parser.parse_args()

    # If start_time is not provided, generate a timestamp
    if args.start_time is None:
        args.start_time = time.strftime("%Y%m%d_%H%M%S")

    return args 



def validate_arguments(args):
    if args.mode == "single_fold" and args.fold is None:
        raise ValueError("--fold is required when mode is 'single_fold'")
    if args.mode == "aggregate" and args.experiment_folder is None:
        raise ValueError("--experiment_folder is required when mode is 'aggregate'")
    

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


def keep_best_trajectories(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
    ):

    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for i, (reward, done) in enumerate(zip(dataset["rewards"], dataset["terminated"])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]
    top_returns = np.array(returns)[top_trajs]
    print(f"mean top returns: {np.mean(top_returns)}")

    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)
    dataset["observations"] = dataset["observations"][order]
    dataset["actions"] = dataset["actions"][order]
    dataset["next_observations"] = dataset["next_observations"][order]
    dataset["rewards"] = dataset["rewards"][order]
    dataset["terminated"] = dataset["terminated"][order]

    return dataset


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            # nn.Tanh(),
            nn.Sigmoid(), # I've change the activation since drugs are scaled to [0, 1]
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
    

class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    
    
    def train(self, batch: TensorBatch) -> Dict[str, float]:


        
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]



def train(dataset, config):
    # set the seed for reproducibility
    set_seed(config.seed, deterministic_torch=True) # this is the same seed used in the data processing for making splits

    # get the best trajectories
    best_trajectories = keep_best_trajectories(dataset.train_data, config.frac, config.discount)

    # corl provides an option to normalize here, but I prenormalized in the data_processor. Further, 
    # I'm not sure there are any reasons not to normalize the data, so I don't require this option. 

    replay_buffer = ReplayBuffer(best_trajectories)
    val_replay_buffer = ReplayBuffer(dataset.val_data)

    # corl provides an option to set a checkpoint path here
    # the checkpoint path should be set by the TrainConfig dataclass
    # with a unique identifier for the run
    # We can implement this later 

    # get the max action from the dataset
    max_action = np.max(np.abs(dataset.train_data["actions"]))

    # set seeds
    seed = config.seed
    set_seed(seed)

    # create the actor
    state_dim = dataset.train_data["observations"].shape[1]
    action_dim = dataset.train_data["actions"].shape[1]
    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    # the training class requires a dictionary of keyword arguments
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Hosp: {config.hospital}, Seed: {seed}")
    print("---------------------------------------")

    trainer = BC(**kwargs)

    # corl code has option to load a checkpoint here
    if config.load_model is not None:
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # create the directory for checkpoints if it doesn't exist
    os.makedirs(config.checkpoints_path, exist_ok=True)

    best_val_loss = float("inf")
    patience = 5  # Number of iterations to wait before stopping
    patience_counter = 0

    num_batches = len(replay_buffer) // config.batch_size

    for e in range(config.n_epochs): # previously max_timesteps
        for batch_idx in range(num_batches):
            if batch_idx % 100 == 0:
                print(f"Epoch {e}, Batch {batch_idx}/{num_batches}")
        
            batch = replay_buffer.sample(config.batch_size)
            # batch = [b.to(config.device) for b in batch] # this line is not needed since the replay buffer returns tensors on the correct device

            log_dict = trainer.train(batch)
            wandb.log(log_dict, step=trainer.total_it)

        if e % config.eval_freq == 0:
            val_batch = val_replay_buffer.sample(config.batch_size)
            with torch.no_grad():
                val_loss = F.mse_loss(trainer.actor(val_batch[0]), val_batch[1]).item()
            wandb.log({"val_loss": val_loss}, step=trainer.total_it)
            print(f"Epoch {e}, Validation Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Optionally save the best model
                save_path = Path(config.checkpoints_path) / "best_model.pt"
                torch.save(trainer.state_dict(), save_path)

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {e} due to no improvement in validation loss.")
                    break
            
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{e}.pt"),
                )

    # Final evaluation on the test set
    test_replay_buffer = ReplayBuffer(dataset.test_data)
    test_batch = test_replay_buffer.sample(config.batch_size)
    with torch.no_grad():
        test_loss = F.mse_loss(trainer.actor(test_batch[0]), test_batch[1]).item()
    print(f"Final Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

    # Save the final model
    final_save_path = Path(config.checkpoints_path) / "final_model.pt"
    torch.save(trainer.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")



# we can use pyrallis to parse command line arguments and load configurations when
# we're further along in the project
def main():

    args = parse_arguments()
    validate_arguments(args)

    # Initialize config manager, it holds paths and environment info
    config_manager = ConfigManager(args)

    if args.mode == "single_fold":
        # create the name and directory for this fold
        config_manager.set_fold_name() # check code, we now now want a parent dir, with inner date dir

        # load data
        dataset = Dataset(config_manager.data_config)
        print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

        # Initialize W&B - will implement later
        # log_fold_results_to_wandb(results)

        # load training config
        config_manager.set_training_config() # based on model type (i.e. any_percent_bc.yaml in this case)   
    
        print(f"Running on {config_manager.environment} environment")
        print(f"Experiment directory: {config_manager.experiment_dir}")
        print(f"Fold directory: {config_manager.fold_dir}")

        results = outer_training_loop(dataset, config_manager)

        # LOG TO W&B in main script
        # log_fold_results_to_wandb(results)

        print(f"Fold {args.fold} completed successfully")
        print(f"Test metrics: {results['test_metrics']}")
        
        if wandb.run is not None:
            wandb.finish()

    



if __name__ == "__main__":
    main()