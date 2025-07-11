from typing import Any, Dict, List, Optional, Tuple, Union

import pyrallis

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import load_config, set_seed
from training_utilities import TrainConfig
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
    dataset["terminals"] = dataset["terminals"][order]

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


def train(dataset: Dict, config: TrainConfig):
    # set the seed for reproducibility
    set_seed(config.seed, deterministic_torch=True) # this is the same seed used in the data processing for making splits

    # get the best trajectories
    best_trajectories = keep_best_trajectories(dataset.train_data, config.frac, config.discount)


    pass

# we can use pyrallis to parse command line arguments and load configurations when
# we're further along in the project
def main():
    
    hospital = 'choc'  # or 'mimic'
    data_config_path = f"config/{hospital}.yaml"
    data_config = load_config(data_config_path)
    processed_data = DataProcessor(data_config)
    
    any_percent_bc_config_path = f"config/any_percent_bc.yaml"
    any_percent_bc_cfg = load_config(any_percent_bc_config_path)
    # add data_config['SEED'] to any_percent_bc_cfg
    any_percent_bc_cfg['seed'] = data_config['seed']
    
    train_cfg = TrainConfig(**any_percent_bc_cfg)  # Convert to TrainConfig dataclass
    print(train_cfg)

    train(processed_data, train_cfg)

    
    

    
    pass



if __name__ == "__main__":
    main()