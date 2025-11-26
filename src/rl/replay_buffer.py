"""
Loads the ICU datasets (currently either MIMIC IV or choc) and stores it in a replay buffer.

Originally, I made my own ReplayBuffer (ReplayBuffer class below) but lab members suggested I 
use pytorch, so I will now build my ReplayBuffer using Dataset and Dataloader.

"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset

TensorBatch = List[torch.Tensor]

ai_clinician_dir = "/home/jake/Documents/AI_Clinician/"


class RLDataset(Dataset):
    def __init__(self, rl_dataset: Dict[str, np.ndarray]):
        self.states = torch.tensor(rl_dataset["observations"], dtype=torch.float32)
        self.actions = torch.tensor(rl_dataset["actions"], dtype=torch.float32)
        self.rewards = torch.tensor(rl_dataset["rewards"], dtype=torch.float32)
        self.next_states = torch.tensor(rl_dataset["next_observations"], dtype=torch.float32)
        # if next actions are included, add them here
        if "next_actions" in rl_dataset:
            self.next_actions = torch.tensor(rl_dataset["next_actions"], dtype=torch.float32)
        self.dones = torch.tensor(rl_dataset["terminated"], dtype=torch.float32)
        self.mortality = torch.tensor(rl_dataset["mortality"], dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if hasattr(self, 'next_actions'):
            return [
                self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.next_actions[idx],
                self.dones[idx],
                self.mortality[idx]
            ]
        else:
            return [
                self.states[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_states[idx],
                self.dones[idx],
                self.mortality[idx]
            ]


class DynamicDeepHitDataset(Dataset):
    def __init__(self, dyn_deephit_dataset: Dict[str, np.ndarray]):
        self.features = torch.tensor(dyn_deephit_dataset["features"], dtype=torch.float32)
        self.time_to_event = torch.tensor(dyn_deephit_dataset["time_to_event"], dtype=torch.float32)
        self.event = torch.tensor(dyn_deephit_dataset["event"], dtype=torch.float32)
        self.mask = torch.tensor(dyn_deephit_dataset["mask"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return [
            self.features[idx],
            self.time_to_event[idx],
            self.event[idx],
            self.mask[idx]
        ]
    

class DynamicDeepHit_Lazy(Dataset):
    """
    For "lazy loading" large datasets that don't fit into memory.
    """
    def __init__(self, index_df, h5_paths, window_size=72):
        self.index = index_df.reset_index(drop=True)
        self.h5_paths = h5_paths
        self.window = window_size
        
        # File handles will be created per worker
        self.files = None

    def _init_files(self):
        """Only open file handles if they haven't been created yet."""
        if self.files is None:
            self.files = [h5py.File(p, "r") for p in self.h5_paths]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        self._init_files()

        row = self.index.iloc[idx]
        
        file = self.files[row.file_id]
        features = file["features"]

        # Random window start
        T = row.length
        start_in_traj = np.random.randint(0, T - self.window + 1)
        end_in_traj = start_in_traj + self.window
        
        # Convert to absolute indices
        abs_start = row.start + start_in_traj
        abs_end = abs_start + self.window
        
        # Lazy load window
        x = features[abs_start:abs_end]     # (window_size, num_features)

        # Compute time to event
        time_to_event = (T - 1) - (end_in_traj - 1)

        event_label = row.event_label

        return (
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(event_label, dtype=torch.long),
            torch.tensor(time_to_event, dtype=torch.long)
        )


def worker_init_fn(worker_id):
    """Initialize each worker with its own file handles."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.files = None




def main():  

    pass

if __name__ == "__main__":
    main()
