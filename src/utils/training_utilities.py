import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    
    # Non-default arguments (required fields, must come first)
    hospital: str
    project: str
    group: str
    experiment: str

    dataframe_path: str
    features_to_use_path: str
    
    model_type: str  # Type of model to train
    seed: int # Sets all seeds
    checkpoints_path: str  # Save path

    targets: List[str] 
    reward: str
    flip_reward: bool

    terminal_reward_col: str
    terminal_reward: bool

    time_col: str
    encounter_col: str

    normalize_obs: bool
    normalize_actions: bool  
    
    # Default arguments (optional fields)
    device: str = "cuda"    
    name: str = ""
    eval_freq: int = 5  # How often (epochs) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    load_model: Optional[str] = None
    n_epochs: int = 1000  # Max time steps to run environment
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    frac: Optional[float] = 0.2  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length

    proportions_to_use: float = 0.90     
   

    def __post_init__(self):
        self.name = f"{self.model_type}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


