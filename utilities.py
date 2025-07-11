import yaml
import os
import random
import numpy as np
import torch



def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)