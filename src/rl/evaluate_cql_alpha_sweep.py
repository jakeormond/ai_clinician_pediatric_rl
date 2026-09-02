import argparse
import json
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import seaborn as sns
import time

import torch
import numpy as np
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .models.td3_bc import Actor as TD3Actor
from .models.cql_with_logprobs import Actor as CQLActor
from .models.cql_with_logprobs import Critic as CQLCritic
from .models.iql import Actor as IQLActor
from .models.iql import Critic as IQLCritic
from .models.iql import ValueNetwork as IQLValue
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..setup_environment import load_paths


def get_model_dirs(base_dir):
    model_dirs = []
    # find directories in base_dir that begin with choc
    choc_dirs = [d for d in os.listdir(base_dir) if d.startswith("choc")]

    if choc_dirs == []:
        print(f"No choc directories found in {base_dir}")
        # look for directories that begin with unknown or local
        sub_dirs = [d for d in os.listdir(base_dir) if d.startswith("unknown") or d.startswith("local")]
        for sub_dir in sub_dirs:
            model_dirs.append(os.path.join(base_dir, sub_dir))
            print(f"Found model directory: {os.path.join(base_dir, sub_dir)}")
    else:
        for choc_dir in choc_dirs:
            # find directories that begin with unknown or local
            sub_dirs = [d for d in os.listdir(os.path.join(base_dir, choc_dir)) if d.startswith("unknown") or d.startswith("local")]
            for sub_dir in sub_dirs:
                model_dirs.append(os.path.join(base_dir, choc_dir, sub_dir))
                print(f"Found model directory: {os.path.join(base_dir, choc_dir, sub_dir)}")
    return model_dirs      