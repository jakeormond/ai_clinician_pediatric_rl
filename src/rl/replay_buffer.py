"""
Loads the ICU datasets (currently either MIMIC IV or choc) and stores it in a replay buffer.

Originally, I made my own ReplayBuffer (ReplayBuffer class below) but lab members suggested I 
use pytorch, so I will now build my ReplayBuffer using Dataset and Dataloader.

"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
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
    







def main():  

    pass

if __name__ == "__main__":
    main()
