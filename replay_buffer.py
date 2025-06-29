"""
Loads the ICU datasets (currently either MIMIC IV or choc) and stores it in a replay buffer.

"""


import numpy as np
import pandas as pd
import torch
import os

ai_clinician_dir = "/home/jake/Documents/AI_Clinician/"


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data icu dataset
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
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

class ReplayBuffer2(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.start_ptr = 0
		self.start_size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.start_state = np.zeros((max_size, state_dim))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def add_start(self, state):
		self.start_state[self.start_ptr] = state

		self.start_ptr = (self.start_ptr + 1) % self.max_size
		self.start_size = min(self.start_size + 1, self.max_size)


	def all_start(self, batch_size=2048):
		if batch_size == -1:
			return torch.FloatTensor(self.start_state[:self.start_size]).to(self.device)
		else:
			ind = np.random.randint(self.start_size, size=batch_size)
			return torch.FloatTensor(self.start_state[ind]).to(self.device)


	def sample(self, batch_size):
		ind = np.random.randint(self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def icu_replay_buffer(hospital = "choc", device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Creates a replay buffer for the ICU dataset.
    :param hospital: str, either "mimic" or "choc"
    :param device: str, either "cuda" or "cpu"
    :return: ReplayBuffer instance

    Currently paths are hardcoded, but paths will be added to the config file in the future
    """
    if hospital == "choc":
        data_path = os.path.join(ai_clinician_dir, "choc_data", 
            "combined_data", "combined_hourly_features_sepsis_proportion_0.80.parquet")
        
        dataset = pd.read_parquet(data_path)
        
    elif hospital == "mimic":
        data_path = os.path.join(ai_clinician_dir, 
            "/mimic_iv_3.1_data/combined_data/hourly/combined_hourly_0.8", 
            "combined_hourly_features.csv.gz")
        
        dataset = pd.read_csv(data_path, compression='gzip')

    # convert the dataset into np arrays of observations, actions, rewards, next_observations, and terminals


def main():  
    
    
      


    pass


if __name__ == "__main__":
    main()
