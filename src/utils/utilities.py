import yaml
import os
import random
import numpy as np
import torch



def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def set_seed(
    seed: int, deterministic_torch: bool = True
):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    # The following ensure complete reproducibility but can slow down training:
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True