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
    

class DynamicDeepHitTransformerDataset(Dataset):
    def __init__(
            self, 
            data_processor,
            config_manager
            ):
        self.trajectories = torch.tensor(data_processor.dynamic_deephit_dataset["trajectories"], dtype=torch.float32)
        self.time_to_event = torch.tensor(data_processor.dynamic_deephit_dataset["time_to_event"], dtype=torch.long)
        self.event = torch.tensor(data_processor.dynamic_deephit_dataset["event"], dtype=torch.long)
        self.mask = torch.tensor(data_processor.dynamic_deephit_dataset["mask"], dtype=torch.bool)
    
    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.trajectories[idx],
            "time_to_event": self.time_to_event[idx],
            "event": self.event[idx],
            "mask": self.mask[idx]
        }
    

class DynamicDeepHit_InMemory(Dataset):
    """
    For datasets that fit into memory.
    """
    def __init__(
            self, 
            data_processor,
            config_manager
            ):
        
        dynamic_deephit_dataset = data_processor.dynamic_deephit_dataset
        self.time_bins = config_manager.time_bins
        self.features = dynamic_deephit_dataset['features']  # (num_windows, window_length, num_features)
        self.mask = dynamic_deephit_dataset['mask']          # (num_windows, window_length)
        self.event = dynamic_deephit_dataset['event']        # (num_windows,)
        self.time_to_event = dynamic_deephit_dataset['time_to_event']  # (num_windows,)
        self.stay_ids = dynamic_deephit_dataset['stay_id']  # (num_windows,)
    
    def __len__(self):
        return len(self.event)

    def __getitem__(self, idx):
        window_feat = torch.from_numpy(self.features[idx]).float()
        mask = torch.from_numpy(self.mask[idx]).float()
        stay_id = self.stay_ids[idx]

        return {
            "features": window_feat,                  # (window_length, F)
            "mask": mask,                             # (window_length,)
            "event": torch.tensor(self.event[idx], dtype=torch.long),
            "time_to_event": torch.tensor(self.time_to_event[idx], dtype=torch.long),
            "stay_id": stay_id                    # useful for debugging
        }


class DynamicDeepHit_InMemory_v1(Dataset):
    """
    For datasets that fit into memory.
    """
    def __init__(
            self, 
            data_processor,
            config_manager
            ):
        
        dynamic_deephit_dataset = data_processor.dynamic_deephit_dataset

        self.feature_names = dynamic_deephit_dataset['feature_names']
        self.stay_ids = dynamic_deephit_dataset['stay_ids']
        self.window_ptr = dynamic_deephit_dataset['window_ptr']
        self.first_window_start_idx = dynamic_deephit_dataset['first_window_start_idx']
        self.trajectories = dynamic_deephit_dataset['trajectories']  # (num_windows, window_length, num_features)

        self.traj_index = data_processor.traj_index_table.set_index('stay_id')
        self.window_index = data_processor.window_index_table
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins
        

    def __len__(self):
        return len(self.window_index)
    
    def __getitem__(self, idx):
        row = self.window_index.iloc[idx]

        stay_id = row.stay_id
        window_start = row.window_start
        window_end = row.window_end  
        start_index = row.start_index
        end_index = row.end_index
        window_length = row.window_length
        time_to_event = row.time_to_event
        event_raw = row.event_raw
        event = event_raw + 1 # convert to 1 or 2 (0 is censored, though we aren't using censored data here)
              
        # load feature slice
        if window_start == window_end:
            # all in one window
            features = self.trajectories[window_start, :, :]
            features = features[start_index:end_index+1]
        else:
            # spans two windows
            first_window = self.trajectories[window_start, :, :]
            second_window = self.trajectories[window_end, :, :]

            first_part = first_window[start_index:]
            second_part = second_window[:end_index+1]

            features = np.vstack([first_part, second_part])

        features = np.asarray(features)   # (slice_len, num_features)
        
        # ---- Pad if needed (repeat first timestep) ----
        if window_length < self.window_length:
            pad = self.window_length - window_length
            first_step = features[0:1]
            pad_block = np.repeat(first_step, pad, axis=0)
            window_feat = np.vstack([pad_block, features])
        else:
            window_feat = features
        
        # Convert to float32 torch tensor
        window_feat = torch.from_numpy(window_feat).float()

        # ---- Mask (actual content only) ----
        mask = torch.ones(self.window_length, dtype=torch.float32)  # this isn't necessary, but needs to 
                                                                    # be removed from model as well.
        # ---- Time-to-event binning ----
        # Right boundary inclusive, then shift to 0-based bins
        tte_bin = np.digitize(time_to_event, self.time_bins, right=True) - 1

        # ---- Package output ----
        return {
            "features": window_feat,                  # (window_length, F)
            "mask": mask,                             # (window_length,)
            "event": torch.tensor(event, dtype=torch.long),
            "time_to_event": torch.tensor(tte_bin, dtype=torch.long),
            "stay_id": stay_id                    # useful for debugging
        }



class DynamicDeepHit_Lazy(Dataset):
    """
    For "lazy loading" large datasets that don't fit into memory.
    """
    def __init__(
            self, 
            data_processor,
            config_manager
            ):
        self.traj_index = data_processor.traj_index_table.set_index('stay_id')
        self.window_index = data_processor.window_index_table
        self.h5_paths = data_processor.normalized_shard_files
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins
        self.shard_dir = data_processor.shard_dir
        
        # Create mapping from file path to index for fast lookup
        self.path_to_idx = {path: i for i, path in enumerate(self.h5_paths)}
        
        # File handles will be created per worker
        self.files = None

    def _init_files(self):
        """Only open file handles if they haven't been created yet."""
        if self.files is None:
            self.files = [h5py.File(path, "r") for path in self.h5_paths]

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        # Make sure H5 files exist in this worker
        self._init_files()

        row = self.window_index.iloc[idx]

        stay_id = row.stay_id
        window_start = row.window_start
        window_end = row.window_end  
        start_index = row.start_index
        end_index = row.end_index
        window_length = row.window_length
        time_to_event = row.time_to_event
        event_raw = row.event_raw
        event = event_raw + 1 # convert to 1 or 2 (0 is censored, though we aren't using censored data here)

        # ---- Retrieve trajectory location ----
        traj_info = self.traj_index.loc[stay_id]
        
        shard_file = traj_info.shard_file
        # Get shard index from path
        shard_idx = self.path_to_idx[os.path.join(self.shard_dir, shard_file)]
               
        # load feature slice
        if window_start == window_end:
            # all in one window
            features = self.files[shard_idx]['trajectories'][window_start, :, :]
            features = features[start_index:end_index+1]
        else:
            # spans two windows
            first_window = self.files[shard_idx]['trajectories'][window_start, :, :]
            second_window = self.files[shard_idx]['trajectories'][window_end, :, :]

            first_part = first_window[start_index:]
            second_part = second_window[:end_index+1]

            features = np.vstack([first_part, second_part])

        features = np.asarray(features)   # (slice_len, num_features)
        
        # ---- Pad if needed (repeat first timestep) ----
        if window_length < self.window_length:
            pad = self.window_length - window_length
            first_step = features[0:1]
            pad_block = np.repeat(first_step, pad, axis=0)
            window_feat = np.vstack([pad_block, features])
        else:
            window_feat = features
        
        # Convert to float32 torch tensor
        window_feat = torch.from_numpy(window_feat).float()

        # ---- Mask (actual content only) ----
        mask = torch.ones(self.window_length, dtype=torch.float32)  # this isn't necessary, but needs to 
                                                                    # be removed from model as well.
        # ---- Time-to-event binning ----
        # Right boundary inclusive, then shift to 0-based bins
        tte_bin = np.digitize(time_to_event, self.time_bins, right=True) - 1

        # ---- Package output ----
        return {
            "features": window_feat,                  # (window_length, F)
            "mask": mask,                             # (window_length,)
            "event": torch.tensor(event, dtype=torch.long),
            "time_to_event": torch.tensor(tte_bin, dtype=torch.long),
            "tte_raw": torch.tensor(time_to_event, dtype=torch.long),
            "stay_id": stay_id                    # useful for debugging
        }
        
        

def worker_init_fn(worker_id):
    """Initialize each worker with its own file handles."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.files = None




def main():  

    pass

if __name__ == "__main__":
    main()
