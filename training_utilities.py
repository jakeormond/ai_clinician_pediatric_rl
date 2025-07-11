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
    model_type: str = None  # Type of model to train
    device: str = "cuda"
    seed: int = None  # Sets all seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    max_timesteps: int = 1e6  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    frac: float = 0.2  # Best data fraction to use
    max_traj_len: int = 1000  # Max trajectory length

   

    def __post_init__(self):
        self.name = f"{self.model_type}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


