from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict
from pathlib import Path
import os

import pyrallis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from utilities import load_config, set_seed
from training_utilities import TrainConfig
from logging_utilities import wandb_init
from data_processor import DataProcessor
from replay_buffer import ReplayBuffer


TensorBatch = List[torch.Tensor]


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
            nn.Tanh(),
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



def train(dataset: Dict, config: TrainConfig):
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

    for t in range(config.max_timesteps):
        batch = replay_buffer.sample(config.batch_size)
        # batch = [b.to(config.device) for b in batch] # this line is not needed since the replay buffer returns tensors on the correct device

        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)

        if t % config.eval_freq == 0:
            val_batch = val_replay_buffer.sample(config.batch_size)
            with torch.no_grad():
                val_loss = F.mse_loss(trainer.actor(val_batch[0]), val_batch[1]).item()
            wandb.log({"val_loss": val_loss}, step=trainer.total_it)
            print(f"Iteration {t}, Validation Loss: {val_loss:.4f}")

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
                    print(f"Early stopping at iteration {t} due to no improvement in validation loss.")
                    break
            
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
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
    
    hospital = 'choc'  # or 'mimic'
    data_config_path = f"config/{hospital}.yaml"
    data_config = load_config(data_config_path)
    processed_data = DataProcessor(data_config, test_size = 0.15, val_size = 0.15)
    
    any_percent_bc_config_path = f"config/any_percent_bc.yaml"
    any_percent_bc_cfg = load_config(any_percent_bc_config_path)
    # merge the data_config into any_percent_bc_cfg
    any_percent_bc_cfg.update(data_config)
    any_percent_bc_cfg['seed'] = data_config['seed']
    any_percent_bc_cfg['hospital'] = data_config['hospital']
    
    train_cfg = TrainConfig(**any_percent_bc_cfg)  # Convert to TrainConfig dataclass
    print(train_cfg)

    train(processed_data, train_cfg)

    



if __name__ == "__main__":
    main()