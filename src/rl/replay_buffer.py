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
    def __init__(self, processor: Dict[str, np.ndarray]):

        self.states = processor.rl_dataset["states"]  # (num_samples, num_features)
        self.actions = processor.rl_dataset["actions"]  # (num_samples, num_actions)
        self.rewards = processor.rl_dataset["rewards"]  # (num_samples,)      
        self.next_states = processor.rl_dataset["next_states"]  # (num_samples, num_features)
        self.next_actions = processor.rl_dataset["next_actions"]  # (num_samples, num_actions)        
        self.dones = processor.rl_dataset["dones"]  # 
        self.mortality = processor.rl_dataset["mortality"]

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


# class DynamicDeepHitDataset(Dataset):
#     def __init__(self, dyn_deephit_dataset: Dict[str, np.ndarray]):
#         self.features = torch.tensor(dyn_deephit_dataset["features"], dtype=torch.float32)
#         self.time_to_event = torch.tensor(dyn_deephit_dataset["time_to_event"], dtype=torch.float32)
#         self.event = torch.tensor(dyn_deephit_dataset["event"], dtype=torch.float32)
#         self.mask = torch.tensor(dyn_deephit_dataset["mask"], dtype=torch.float32)

#     def __len__(self) -> int:
#         return len(self.mask)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         return [
#             self.features[idx],
#             self.time_to_event[idx],
#             self.event[idx],
#             self.mask[idx]
#         ]
    


class DynamicDeepHitTransformerDataset(Dataset):
    """
    self.reshuffle_subsample() should be called at the start of each epoch to reshuffle and select new windows.
    """

    def __init__(
            self, 
            data_processor,
            config_manager,
            ):
        self.shuffle_inputs = config_manager.training_config.get('shuffle_inputs', False)
        
        self.trajectories = torch.tensor(data_processor.dynamic_deephit_dataset["trajectories"], dtype=torch.float32)
        self.feature_names = data_processor.dynamic_deephit_dataset["feature_names"]
        self.stay_ids = torch.tensor(data_processor.dynamic_deephit_dataset["stay_ids"], dtype=torch.long)
        self.time_to_event_raw = torch.tensor(data_processor.dynamic_deephit_dataset["time_to_event_raw"], dtype=torch.long)
        self.time_to_event = torch.tensor(data_processor.dynamic_deephit_dataset["time_to_event"], dtype=torch.long)
        self.event = torch.tensor(data_processor.dynamic_deephit_dataset["event"], dtype=torch.long)
        self.index_ptr = torch.tensor(data_processor.dynamic_deephit_dataset["index_ptr"], dtype=torch.long)
        self.window_length = config_manager.training_config['window_length']
        self.mode = data_processor.split # 'train', 'val', or 'test' - train is random windows, val/test is deterministic windows
        
        # each timestep is the end of a potential window
        self.n_windows = self.trajectories.shape[0]

        # for an epoch, we will sample self.n_windows/window_length windows
        self.n_samples = self.n_windows // self.window_length

        self.features_to_use = pd.read_csv(config_manager.data_config['features_to_use_path'])
        
        # get indices of response features to zero out or carry forward at the final timestep (for RL setting)
        self.response_zero_ind = data_processor.dynamic_deephit_dataset['response_zero_ind']
        self.carry_forward_ind = data_processor.dynamic_deephit_dataset['carry_forward_ind']

        # get the indices of the actions to be used for RL, it is the intersection of these
        # and the response_zero_ind and carry_forward_ind features which we need to omit 
        # from the final timestep
        actions = config_manager.training_config['actions']
        # if actions is a list of lists, flatten it
        if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], list):
            actions = [item for sublist in actions for item in sublist]
               
        if actions is not None:
            action_ind = []
            for action in actions:
                ind = self.feature_names.index(action)
                action_ind.append(ind)

                mask_feature_name = action + "_mask"
                if mask_feature_name in self.feature_names:                
                    ind2 = self.feature_names.index(action + "_mask")
                    action_ind.append(ind2)

                delta_feature_name = action + "_delta"
                if delta_feature_name in self.feature_names:
                    ind3 = self.feature_names.index(action + "_delta")
                    action_ind.append(ind3)


            self.response_zero_ind = np.sort(list(set(self.response_zero_ind) & set(action_ind)))
            self.carry_forward_ind = np.sort(list(set(self.carry_forward_ind) & set(action_ind)))

        self.reshuffle_subsample() 


    def reshuffle_subsample(self):
        """shuffle self.window_end_indices and select first n_samples indices"""
        # shuffle self.window_end_indices
        perm = torch.randperm(self.n_windows)
        # sample first n_samples indices
        perm = perm[:self.n_samples]
        self.window_end_indices = perm
        

    def __len__(self) -> int:
        return self.n_samples   

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        final_ind = self.window_end_indices[idx]
        # get the index of the last self.index_ptr that is less than or equal to final_ind
        start_ind = torch.where(self.index_ptr <= final_ind)[0][-1]
        
        # get the final self.index_ptr value that is equal or less than final_ind
        start_ptr = torch.max(self.index_ptr[start_ind])
        start_to_end_len = final_ind - start_ptr + 1

        # if start_to_end_len > self.window_length, make start_ind equal to final_ind - window_length + 1
        if start_to_end_len > self.window_length:
            start_ptr = final_ind - self.window_length + 1
            start_to_end_len = self.window_length

        # ---- Load trajectory slice ----
        trajectories = self.trajectories[start_ptr:final_ind+1]  # (slice_len, num_features)
        # load time_to_event and event at final_ind
        time_to_event = self.time_to_event[start_ptr:final_ind+1]        
        event = self.event[start_ind]
        padding_mask = torch.zeros(self.window_length, dtype=torch.bool)
        padding_mask[-start_to_end_len:] = 1  # mask for actual content only        

        if self.shuffle_inputs:
            # shuffle time_to_event
            time_to_event = time_to_event[torch.randperm(len(time_to_event))]

        # ---- Pad if needed (repeat first timestep) ----
        if start_to_end_len < self.window_length:
            pad = self.window_length - start_to_end_len
            first_step = trajectories[0:1]
            pad_block = first_step.repeat(pad, 1)
            trajectories = torch.cat([pad_block, trajectories], dim=0)
            tte_pad_block = time_to_event[0:1].repeat(pad)
            time_to_event = torch.cat([tte_pad_block, time_to_event], dim=0)

        # zero out or carry forward response features at final timestep for RL setting
        if len(self.response_zero_ind) > 0:
            trajectories[-1, self.response_zero_ind] = 0.0
        if len(self.carry_forward_ind) > 0:
            trajectories[-1, self.carry_forward_ind] = trajectories[-2, self.carry_forward_ind]

        # add local window time feature (time since start of window) as an additional input feature
        local_time = torch.arange(self.window_length, dtype=torch.float32).unsqueeze(1)  # (window_length, 1)
        # if padding was added, repeat the local time for the first timestep in the padded region
        if start_to_end_len < self.window_length:
            local_time[:pad] = local_time[pad]  # repeat the local time for the first timestep in the padded region

        # min max the local time
        local_time = local_time / (self.window_length - 1)  # scale to [0, 1]

        trajectories = torch.cat([trajectories, local_time], dim=1)  # (window_length, num_features + 1)
    
        return {
            "features": trajectories,
            "time_to_event": time_to_event,
            "event": event + 1,  # convert to 1 or 2 (0 is censored, though we aren't using censored data here)
            "loss_mask": padding_mask
        }


class DynamicDeepHitTransformerDataset_v1(Dataset):
    def __init__(
            self, 
            data_processor,
            config_manager
            ):
        self.trajectories = torch.tensor(data_processor.dynamic_deephit_dataset["trajectories"], dtype=torch.float32)
        self.time_to_event = torch.tensor(data_processor.dynamic_deephit_dataset["time_to_event"], dtype=torch.long)
        self.event = torch.tensor(data_processor.dynamic_deephit_dataset["event"], dtype=torch.long)
        # self.mask = torch.tensor(data_processor.dynamic_deephit_dataset["mask"], dtype=torch.bool)
        self.shuffle_inputs = config_manager.training_config.get('shuffle_inputs', False)
        self.zero_inputs = config_manager.training_config.get('zero_inputs', False)

    def __len__(self) -> int:
        return len(self.mask)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.shuffle_inputs:
            # get the mask for this sample
            mask = self.mask[idx]  # (window_length,)
            traj = self.trajectories[idx]  # (window_length, num_features)
            # only shuffle the unmasked timesteps
            unmasked_indices = torch.where(~mask)[0]
            shuffled_indices = unmasked_indices[torch.randperm(len(unmasked_indices))]
            shuffled_traj = traj.clone()
            shuffled_traj[unmasked_indices] = traj[shuffled_indices]

            return {
                "features": shuffled_traj,
                "time_to_event": self.time_to_event[idx],
                "event": self.event[idx],
                "mask": self.mask[idx]
            }

        elif self.zero_inputs:
            traj = self.trajectories[idx]  # (window_length, num_features)
            zeroed_traj = traj.clone()
            zeroed_traj[~self.mask[idx]] = 0.0

            # make the mask all False
            zeroed_mask = torch.zeros_like(self.mask[idx], dtype=torch.bool)

            return {
                "features": zeroed_traj,
                "time_to_event": self.time_to_event[idx],
                "event": self.event[idx],
                "mask": zeroed_mask
            }

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
