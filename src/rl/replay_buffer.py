"""
Loads the ICU datasets (currently either MIMIC IV or choc) and stores it in a replay buffer.

"""
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch



TensorBatch = List[torch.Tensor]

ai_clinician_dir = "/home/jake/Documents/AI_Clinician/"


class ReplayBuffer:
    def __init__(
        self,
        rl_dataset: Dict[str, np.ndarray], # should contain keys: 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._device = device

        self._states = self._to_tensor(rl_dataset["observations"])
        self._actions = self._to_tensor(rl_dataset["actions"])
        self._rewards = self._to_tensor(rl_dataset["rewards"])
        self._next_states = self._to_tensor(rl_dataset["next_observations"])
        self._dones = self._to_tensor(rl_dataset["terminated"])
        self.size = self._states.shape[0]   

        # n transitions is the number of rows in the dataset minus 
        # the number of dones
        self.n_transitions = self.size - self._dones.sum().item()   

        print(f"Replay buffer initialized with {self.n_transitions} transitions.")

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data icu dataset - IS THIS UNUSED?????????????
    def load_icu_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions

        print(f"Dataset size: {n_transitions}")

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self.size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]


def main():  

    pass

if __name__ == "__main__":
    main()
