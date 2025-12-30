"""
Creates DataProcessor class to process data stored in HDF5 format.
All normalization and transformation is done here.
"""
import yaml
import copy
import h5py
import os
import glob
import pandas as pd
import numpy as np
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

import torch
from torch.utils.data import Dataset, DataLoader

import logging

from src import config_manager
logger = logging.getLogger(__name__)

class ShardProcessor:
    def __init__(
            self, 
            config_manager, 
            split = 'train', 
            normalize = True,
            norm_params=None, 
        ):
        """
        Unlike with dataframes, with the shards, I've already dropped the features we
        don't want to use, so we just need to normalize, create binary actions,
        and then split into the appropriate dataset format.
        Note that features have already been transformed as well
        -------------
        
        """
        self.split = split
        self.norm_params = norm_params
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins
        self.shuffle_inputs = config_manager.training_config.get('shuffle_inputs', False)

        # load features to use from config_manager
        self.features_to_use = pd.read_csv(config_manager.data_config['features_to_use_path'])

        
        # if config_manager.action_mode == 'binary_only' or config_manager.action_mode == 'mixed':
        #     create_binary_actions = True
        # else:
        #     create_binary_actions = False            

        # if config_manager.action_mode == 'binary_only':
        #     self.normalize_actions = False
        #     self.transform_actions = False           

        if self.split == 'train' and config_manager.training_config['tuning']:
            self.split = 'tuning'

        # find the shards for the given split
        if not config_manager.training_config['tuning']:
            self.shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'], 'training_set')
        else:
            self.shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'], 'tuning_set')

        all_files = os.listdir(self.shard_dir)                
        
        raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith("_shard.h5")]
        # sort the files to ensure consistent order
        self.raw_shard_files = sorted(raw_shard_files)

        if normalize:
            # normalized shard files are different for standard and transformer based dynamic deephit
            if config_manager.model_type == "dynamic_deephit_transformer":
                normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                    f.startswith(self.split) and f.endswith("_shard_normalized_transformer.h5")]
            else:
                normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                        f.startswith(self.split) and f.endswith("_shard_normalized.h5")]
            # sort the files to ensure consistent order
            self.normalized_shard_files = sorted(normalized_shard_files)

            if len(self.normalized_shard_files) == 0: # i.e. the files have not been normalized yet
                self.normalize(config_manager=config_manager)
                all_files = os.listdir(self.shard_dir)
                if config_manager.model_type == "dynamic_deephit_transformer":
                    normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                        f.startswith(self.split) and f.endswith("_shard_normalized_transformer.h5")]
                else:
                    normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                        f.startswith(self.split) and f.endswith("_shard_normalized.h5")]
                
                # sort the files to ensure consistent order
                self.normalized_shard_files = sorted(normalized_shard_files)

        else:
            # raise NotImplementedError("Non-normalized shard processing not implemented yet.")
            raise NotImplementedError()
        
        # load the normalized shard files and store in a dynamic_deephit_dataset attribute
        if config_manager.model_type == "dynamic_deephit_transformer":
            index_type = 'window_end_idx'
        else:
            index_type = 'first_window_start_idx'
        
        self.dynamic_deephit_dataset = {}
        for i, shard_file in enumerate(self.normalized_shard_files):
            logger.info(f"Loading normalized shard file into memory for tuning: {shard_file}")
            # load the shard file
            with h5py.File(shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                stay_ids = f['stay_ids'][:]
                window_ptr = f['window_ptr'][:]
                window_start_or_end_idx = f[index_type][:]
                trajectories = f['trajectories'][:]             

            if i == 0:
                self.dynamic_deephit_dataset['feature_names'] = [name.encode('utf-8') for name in feature_names]
                self.dynamic_deephit_dataset['stay_ids'] = stay_ids
                self.dynamic_deephit_dataset['window_ptr'] = window_ptr
                self.dynamic_deephit_dataset[index_type] = window_start_or_end_idx
                self.dynamic_deephit_dataset['trajectories'] = trajectories.astype(np.float32)
            else:
                self.dynamic_deephit_dataset['stay_ids'] = np.concatenate(
                    (self.dynamic_deephit_dataset['stay_ids'], stay_ids), axis=0)
                self.dynamic_deephit_dataset['window_ptr'] = np.concatenate(
                    (self.dynamic_deephit_dataset['window_ptr'], 
                        window_ptr[1:] + self.dynamic_deephit_dataset['window_ptr'][-1]), axis=0)
                self.dynamic_deephit_dataset[index_type] = np.concatenate(
                    (self.dynamic_deephit_dataset[index_type], window_start_or_end_idx), axis=0)
                self.dynamic_deephit_dataset['trajectories'] = np.concatenate(
                    (self.dynamic_deephit_dataset['trajectories'], trajectories.astype(np.float32)), axis=0)


        # get physiological feature indices for dynamic deephit model, and feature_dim
        # load 'feature_names' from the first h5 file
        with h5py.File(self.normalized_shard_files[0], 'r') as f:
            feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))] 
            self.feature_dim = f['trajectories'].shape[2]

        self.physiological_feature_ind = [] # for dynamic deephit model, which tries to predict next physiological features
        self.physiological_feature_type = []

        physiological_feature_sets = [
            'hemodynamics', 'labs', 'neurological', 'routine_vital_signs'
            ]
        
        for i, feature in enumerate(feature_names):
            feature_row = self.features_to_use[self.features_to_use['feature'] == feature]            
            feature_set = feature_row['feature_set'].values[0]
            data_type = feature_row['data_type'].values[0]
            if feature_set in physiological_feature_sets:
                self.physiological_feature_ind.append(i)
                self.physiological_feature_type.append(data_type)

        # create or load index table
        if config_manager.model_type == "dynamic_deephit_transformer":
            self.index_table_path = os.path.join(self.shard_dir, 
                f"{self.split}_index_table_transformer_{self.window_length}.parquet")
        else:
            self.index_table_path = os.path.join(self.shard_dir, f"{self.split}_index_table.parquet")
        if os.path.exists(self.index_table_path):
            logger.info(f"Loading existing index table from {self.index_table_path}")
            self.traj_index_table = pd.read_parquet(self.index_table_path)
        else:
            # create index table
            self.create_index_table(config_manager)
        
        if config_manager.experiment_variant == 'prediction':
            self.create_index_table_for_all_windows(config_manager)           
        elif config_manager.model_type == "dynamic_deephit_transformer":
            self.create_index_table_for_transformer_windows(config_manager)
        else:
            self.create_index_table_for_window_sampling(config_manager)

        if config_manager.model_type == "dynamic_deephit_transformer":
            # we will add to the dynamic_deephit_dataset the events, time_to_events, and mask here
            # loop throught the window_index_table and extract the relevant data
            n_windows = len(self.window_index_table)
            
            # drop the window_ptr, window_end_idx from the dynamic_deephit_dataset to save memory
            del self.dynamic_deephit_dataset['window_ptr']
            del self.dynamic_deephit_dataset['window_end_idx']

            # get the window indices
            window_indices = self.window_index_table['window_idx'].values
            
            # drop any windows from self.dynamic_deephit_dataset['trajectories'] that are not in window_indices
            self.dynamic_deephit_dataset['trajectories'] = \
                self.dynamic_deephit_dataset['trajectories'][window_indices]               

            # calulate mask, event, and time_to_event for each timstep for the transformer model
            event_per_window = self.window_index_table['event_raw'].values + 1  # convert to 1 for survival, 2 for death
            # even_per_window needs to be repeated at every timestep in the window, with final shape (n_windows, window_length)
            self.dynamic_deephit_dataset['event'] = np.repeat(event_per_window[:, np.newaxis], self.window_length, axis=1).astype(np.int32)
            # initialize time_to_event and mask as arrays of shape (n_windows, window_length)
            self.dynamic_deephit_dataset['time_to_event'] = np.zeros((n_windows, self.window_length), dtype=np.int32)
            self.dynamic_deephit_dataset['mask'] = np.zeros((n_windows, self.window_length), dtype=np.int32)

            # time binning here !!!!!!!!!!!!!!!!!!!!!
            for i, row in self.window_index_table.iterrows():
                time_to_event = np.arange(row['time_to_event'] + row['sample_end_idx'], row['time_to_event']-1, -1)
                tte_bin = np.digitize(time_to_event, self.time_bins, right=True) - 1                
                self.dynamic_deephit_dataset['time_to_event'][i, :row['sample_end_idx'] + 1] = tte_bin
                if row['sample_end_idx'] < self.window_length - 1:                
                    self.dynamic_deephit_dataset['mask'][i, row['sample_end_idx'] + 1:] = 1
                
        elif config_manager.training_config['use_windows_from_memory']: # not necessary for transformer model as 
            # everything is already in memory in the self.dynamic_deephit_dataset
            # !!! but the time binning is happening here, so I need to implement it separately for transformer model,
            # probably in the dataloader !!!!!!!!!!! 
            logger.info("Using windows loaded into memory for training.")

            self.window_file_path = os.path.join(self.shard_dir, f"{self.split}_windows.h5")

            if not os.path.exists(self.window_file_path):
                # create the file 
                self.create_windows()

            else:
                # load the windows from file
                self.load_windows()

            # clear the dynamic_deephit_dataset to save memory
            self.dynamic_deephit_dataset = {}

            # load the precomputed windows into self.dynamic_deephit_dataset
            with h5py.File(self.window_file_path, 'r') as f:
                self.dynamic_deephit_dataset['features'] = f['features'][:]
                self.dynamic_deephit_dataset['mask'] = f['mask'][:]
                self.dynamic_deephit_dataset['event'] = f['event'][:]
                self.dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:]
                self.dynamic_deephit_dataset['stay_id'] = f['stay_id'][:]


    def create_index_table(self, config_manager):
        # create an index table mapping from global index to (shard_file, local_index)
        for i, file_path in enumerate(self.raw_shard_files):
            # get death feature and lengths from the raw files
            with h5py.File(file_path, 'r') as f:
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                    
                stay_ids = f['stay_ids'][:]
                index_ptr = f['index_ptr'][:]
                trajectories = f['trajectories'][:]     

            # get events, which are the "death" feature in the final row of each trajectory
            death_feature_index = feature_names.index('death')
            end_indices = index_ptr[1:] - 1
            events = trajectories[end_indices, death_feature_index].astype(int)
            lengths = np.diff(index_ptr).astype(int)

            # then open the normalized file to get the correct indices
            file_path = self.normalized_shard_files[i]
            if config_manager.model_type == "dynamic_deephit_transformer":
                index_type = 'window_end_idx'
            else:
                index_type = 'first_window_start_idx'
            
            with h5py.File(file_path, 'r') as f:
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]
                stay_ids = f['stay_ids'][:]
                window_ptr = f['window_ptr'][:]
                start_or_end_idx = f[index_type][:]
                trajectories = f['trajectories'][:]

            rows = []
            file_name = os.path.basename(file_path)
            for stay_id, start_window, start_or_end_idx, length, event in \
                zip(stay_ids, window_ptr[:-1], start_or_end_idx, lengths, events):
                rows.append({
                    'stay_id': stay_id,
                    'shard_file': file_name,
                    'window_idx': start_window,
                    f"{index_type}": start_or_end_idx, # first window
                    'length': length,
                    'event': event
                })
            
            if i == 0:
                df = pd.DataFrame(rows)
            else:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

        self.traj_index_table = df

        # save the index table to the shard dir
        if config_manager.model_type == "dynamic_deephit_transformer":
            index_table_path = os.path.join(self.shard_dir, f"{self.split}_index_table_transformer.parquet")
        else:
            index_table_path = os.path.join(self.shard_dir, f"{self.split}_index_table.parquet")
        df.to_parquet(index_table_path)
        logger.info(f"Saved index table to {index_table_path}")
   
    
    def create_index_table_for_transformer_windows(self, config_manager):
        """
        For transformer based dynamic deephit model, we need to create an index table for 
        all windows in the dataset.
        """
        window_index_table_path = os.path.join(self.shard_dir, 
            f"{self.split}_all_windows_index_table_transformer_{self.window_length}.parquet")
        
        if os.path.exists(window_index_table_path):
            self.window_index_table = pd.read_parquet(window_index_table_path) 
            logger.info(f"loaded window index table from {window_index_table_path}")
            return
        
        rows = []

        for _, row in self.traj_index_table.iterrows():
            stay_id = row['stay_id']
            traj_length = row['length']
            first_window_idx = row['window_idx']
            end_idx = row['window_end_idx']
        
            if traj_length > config_manager.max_encounter_length:
                continue

            n_windows = int(np.ceil(traj_length / self.window_length))

            event_raw = row['event'] # will convert to 1 for survival, 2 for death to be consistent with existing code
            time_to_event = np.arange(traj_length, 0, -1)

            for window in range(n_windows):        
                if window == n_windows-1:
                    sample_end_idx = end_idx
                    tte_index = end_idx + window * self.window_length
                else:
                    sample_end_idx = self.window_length - 1
                    tte_index = ((window + 1) * self.window_length) - 1
                    
                tte = time_to_event[tte_index]
                window_idx = first_window_idx + window

                rows.append({
                    "stay_id": stay_id,
                    "window_idx": int(window_idx),                     # window index within trajectory
                    "sample_end_idx": int(sample_end_idx),         # end index (0-based)
                    "time_to_event": int(tte), # from final timestep
                    "event_raw": event_raw
                })

        # build dataframe
        window_index_table = pd.DataFrame(rows)

        # sort by stay_id and window_end   
        self.window_index_table = window_index_table.sort_values(by=['stay_id', 'window_idx']).reset_index(drop=True)

        # save the window index table to the shard dir
        self.window_index_table.to_parquet(window_index_table_path)
        logger.info(f"Saved window index table to {window_index_table_path}")


    def create_index_table_for_all_windows(self, config_manager):
        """
        For generating predictions at all time points, we need to create an index table for 
        all windows in the dataset.
        """

        window_index_table_path = os.path.join(self.shard_dir, f"{self.split}_all_windows_index_table.parquet")
        
        if os.path.exists(window_index_table_path):
            self.window_index_table = pd.read_parquet(window_index_table_path) 
            logger.info(f"loaded window index table from {window_index_table_path}")
            return
        
        rows = []

        for _, row in self.traj_index_table.iterrows():
            stay_id = row['stay_id']
            traj_length = row['length']
            window_idx = row['window_idx']
            start_idx = row['start_index']
        
            if traj_length > config_manager.max_encounter_length:
                continue

            event_raw = row['event'] # will convert to 1 for survival, 2 for death to be consistent with existing code
            time_to_event = np.arange(traj_length, 0, -1)

            for idx in range(len(time_to_event)):                
                
                tte = time_to_event[idx]   # identical to your logic
                sample_end_idx = start_idx + idx
                sample_end_window_idx = np.floor(sample_end_idx/self.window_length)

                if sample_end_window_idx > 0:
                    window_length = self.window_length
                    sample_end_idx = sample_end_idx - (sample_end_window_idx * self.window_length)
                    sample_start_idx = sample_end_idx - self.window_length + 1
                    
                    if sample_start_idx < 0:
                        sample_start_window_idx = sample_end_window_idx - 1
                        sample_start_idx = self.window_length + sample_start_idx
                    else:
                        sample_start_window_idx = sample_end_window_idx

                else:
                    sample_start_window_idx = 0
                    sample_start_idx = 0 
                    window_length = sample_end_idx + 1               
            
                sample_start_window_idx += window_idx
                sample_end_window_idx += window_idx

                rows.append({
                    "stay_id": stay_id,
                    "window_start": int(sample_start_window_idx), # start index (0-based)
                    "window_end": int(sample_end_window_idx),         # end index (0-based)
                    "start_index": int(sample_start_idx),         # start index within window
                    "end_index": int(sample_end_idx),             # end index within window
                    "window_length": int(window_length),
                    "time_to_event": int(tte),
                    "event_raw": event_raw
                })

        # build dataframe
        window_index_table = pd.DataFrame(rows)

        # sort by stay_id and window_end   
        self.window_index_table = window_index_table.sort_values(by=['stay_id', 'window_end', 'end_index']).reset_index(drop=True)

        # save the window index table to the shard dir
        self.window_index_table.to_parquet(window_index_table_path)
        logger.info(f"Saved window index table to {window_index_table_path}")


    def create_index_table_for_window_sampling(self, config_manager):
        
        window_index_table_path = os.path.join(self.shard_dir, f"{self.split}_window_index_table.parquet")
        # if file already exists, load it
        if os.path.exists(window_index_table_path):
            self.window_index_table = pd.read_parquet(window_index_table_path) 
            logger.info(f"loaded window index table from {window_index_table_path}")
            return

        seed = config_manager.training_config.get('random_seed', 42)
        rng = np.random.default_rng(seed)
        
        rows = []

        min_gap = int(np.floor(np.min(np.diff(self.time_bins))))
        min_gap = max(min_gap, 1)
        n_windows_per_bin = max(int(min_gap / 4), 4)

        for _, row in self.traj_index_table.iterrows():
            stay_id = row['stay_id']
            traj_length = row['length']
            window_idx = row['window_idx']
            start_idx = row['start_index']

            if traj_length > config_manager.max_encounter_length:
                continue

            event_raw = row['event'] # will convert to 1 for survival, 2 for death to be consistent with existing code

            time_to_event = np.arange(traj_length, 0, -1)

            for b in range(len(self.time_bins) - 1):
                lower_bound = self.time_bins[b]
                upper_bound = self.time_bins[b+1]

                indices_in_bin = np.where((time_to_event <= upper_bound) & 
                                          (time_to_event > lower_bound))[0]
                
                if len(indices_in_bin) == 0:
                    continue

                n_to_sample = min(n_windows_per_bin, len(indices_in_bin))
                sampled = rng.choice(indices_in_bin, size=n_to_sample, replace=False)

                for idx in sampled:
                    tte = time_to_event[idx]   # identical to your logic
                    sample_end_idx = start_idx + idx
                    sample_end_window_idx = np.floor(sample_end_idx/self.window_length)
                    
                    if sample_end_window_idx > 0:
                        window_length = self.window_length
                        sample_end_idx = sample_end_idx - (sample_end_window_idx * self.window_length)
                        sample_start_idx = sample_end_idx - self.window_length + 1
                        
                        if sample_start_idx < 0:
                            sample_start_window_idx = sample_end_window_idx - 1
                            sample_start_idx = self.window_length + sample_start_idx
                        else:
                            sample_start_window_idx = sample_end_window_idx

                    else:
                        sample_start_window_idx = 0
                        sample_start_idx = 0 
                        window_length = sample_end_idx + 1               
             
                    sample_start_window_idx += window_idx
                    sample_end_window_idx += window_idx

                    rows.append({
                        "stay_id": stay_id,
                        "window_start": int(sample_start_window_idx), # start index (0-based)
                        "window_end": int(sample_end_window_idx),         # end index (0-based)
                        "start_index": int(sample_start_idx),         # start index within window
                        "end_index": int(sample_end_idx),             # end index within window
                        "window_length": int(window_length),
                        "time_to_event": int(tte),
                        "event_raw": event_raw
                    })

        # build dataframe
        window_index_table = pd.DataFrame(rows)

        # sort by stay_id and window_end   
        self.window_index_table = window_index_table.sort_values(by=['stay_id', 'window_end', 'end_index']).reset_index(drop=True)

        # save the window index table to the shard dir
        self.window_index_table.to_parquet(window_index_table_path)
        logger.info(f"Saved window index table to {window_index_table_path}")


    def create_windows(self):
        """
        Create windows from the normalized shard files and save to new h5 file
        """
        # iterate through self.window_index_table and extract the windows from the appropriate shard files
        features_all = []
        mask_all = []
        event_all = []
        time_to_event_all = []
        stay_id_all = []
        
        # convert self.traj_index_table index to stay_id for faster lookup
        self.traj_index_table = self.traj_index_table.set_index('stay_id')

        for i, row in self.window_index_table.iterrows():
            stay_id = row.stay_id
            window_start = row.window_start
            window_end = row.window_end  
            start_index = row.start_index
            end_index = row.end_index
            window_length = row.window_length
            time_to_event = row.time_to_event
            event_raw = row.event_raw
            event = event_raw + 1 # con

            # ---- Retrieve trajectory location ----
            traj_info = self.traj_index_table.loc[stay_id]

            # load feature slice
            if window_start == window_end:
                # all in one window
                features = self.dynamic_deephit_dataset['trajectories'][window_start, :, :]
                features = features[start_index:end_index+1]
            else:
                # spans two windows
                first_window = self.dynamic_deephit_dataset['trajectories'][window_start, :, :]
                second_window = self.dynamic_deephit_dataset['trajectories'][window_end, :, :]

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

            # add to lists
            features_all.append(window_feat)
            mask_all.append(mask)
            event_all.append(event)
            time_to_event_all.append(tte_bin)
            stay_id_all.append(stay_id)

        # concatenate all data to np arrays and save to h5 file
        features_all = np.stack(features_all, axis=0).astype(np.float32)
        mask_all = np.stack(mask_all, axis=0).astype(np.float32)
        event_all = np.array(event_all, dtype=np.int32)
        time_to_event_all = np.array(time_to_event_all, dtype=np.int32)
        stay_id_all = np.array(stay_id_all, dtype=np.int32)

        with h5py.File(self.window_file_path, 'w') as f:
            f.create_dataset('features', data=features_all, dtype=np.float32, 
                             chunks=(1, self.window_length, features_all.shape[2]))
            f.create_dataset('mask', data=mask_all, dtype=np.float32, 
                             chunks=(1, self.window_length))
            f.create_dataset('event', data=event_all, dtype=np.int32)
            f.create_dataset('time_to_event', data=time_to_event_all, dtype=np.int32)
            f.create_dataset('stay_id', data=stay_id_all, dtype=np.int32)


    def load_windows(self):
        """
        Load windows from precomputed h5 file
        """
        self.window_file_path = os.path.join(self.shard_dir, f"{self.split}_windows.h5")

        if not os.path.exists(self.window_file_path):
            raise FileNotFoundError(f"Window file {self.window_file_path} not found.")

        logger.info(f"Loading windows from {self.window_file_path}")
        
        with h5py.File(self.window_file_path, 'r') as f:
            self.dynamic_deephit_dataset['features'] = f['features'][:]
            self.dynamic_deephit_dataset['mask'] = f['mask'][:]
            self.dynamic_deephit_dataset['event'] = f['event'][:]
            self.dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:]
            self.dynamic_deephit_dataset['stay_id'] = f['stay_id'][:]

    def normalize(self, config_manager=config_manager):

        if self.norm_params is not None:
            # if normalization parameters are provided, use them to normalize the data
            self.obs_norm_dict = self.norm_params['obs_norm_dict']
            self.action_minimum = self.norm_params['action_minimum']
            self.action_maximum = self.norm_params['action_maximum']
            self.action_transformers = self.norm_params['action_transformers']

        else:
            self.obs_norm_dict = None
            self.action_minimum = None
            self.action_maximum = None
            self.action_transformers = None

            for i, shard_file in enumerate(self.raw_shard_files):
                logger.info(f"Normalizing shard file: {shard_file}")
                # load the shard file
                with h5py.File(shard_file, 'r') as f:      
                    # convert feature_names from bytes to strings
                    feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                    stay_ids = f['stay_ids'][:]
                    index_ptr = f['index_ptr'][:]
                    trajectories = f['trajectories'][:]             
                    
                    # get the index of the death feature, and drop it from the trajectories
                    death_feature_index = feature_names.index('death')
                    # update feature_names
                    feature_names.pop(death_feature_index)
                    trajectories = np.delete(trajectories, death_feature_index, axis=1)
                    
                    if i == 0:
                        n_features = len(feature_names)

                        sums = np.zeros(n_features)
                        sq_sums = np.zeros(n_features)
                        counts = 0
                        minima = np.full(n_features, np.inf)
                        maxima = np.full(n_features, -np.inf)

                    # compute sums, squared sums, counts, minima, maxima for 1 feature at a time to avoid floating point catastophic cancellation
                    for j in range(n_features):
                        feature_trajectories = trajectories[:, j]
                        sums[j] += np.sum(feature_trajectories)
                        sq_sums[j] += np.sum(feature_trajectories ** 2)
                        minima[j] = np.minimum(minima[j], np.min(feature_trajectories))
                        maxima[j] = np.maximum(maxima[j], np.max(feature_trajectories))
                    counts += trajectories.shape[0]

            means = sums / counts
            variances = (sq_sums / counts) - (means ** 2)
            # elementwise clamp variances at 0 to avoid tiny negative values from numerical precision
            stds = np.sqrt(np.maximum(variances, 0.0))  # avoid negative variance due to numerical issues

            self.obs_norm_dict = {}
            for i, feature in enumerate(feature_names):
                self.obs_norm_dict[feature] = {
                    'min': minima[i],
                    'max': maxima[i],
                    'mean': means[i],
                    'std': stds[i]
                }

        # now normalize each shard file and save to new file
        for i, shard_file in enumerate(self.raw_shard_files):
            logger.info(f"Normalizing shard file: {shard_file}")
            # load the shard file
            with h5py.File(shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                stay_ids = f['stay_ids'][:]
                index_ptr = f['index_ptr'][:]
                trajectories = f['trajectories'][:]

                # get the index of the death feature, and drop it from the trajectories
                death_feature_index = feature_names.index('death')
                # update feature_names
                feature_names.pop(death_feature_index)
                n_features = len(feature_names)
                trajectories = np.delete(trajectories, death_feature_index, axis=1)

                # normalize the trajectories
                for j, feature in enumerate(feature_names):
                    feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
                    data_type = feature_row['data_type'].values[0]
                    feature_set = feature_row['feature_set'].values[0]

                    if data_type == 'ordinal' or feature_set == 'medications':
                        # min-max normalize the feature
                        max_min_diff = self.obs_norm_dict[feature]['max'] - self.obs_norm_dict[feature]['min']
                        trajectories[:, j] = (trajectories[:, j] - self.obs_norm_dict[feature]['min']) / max_min_diff  
                    
                    elif data_type == 'continuous':
                        # z-score normalize the feature
                        trajectories[:, j] = (trajectories[:, j] - self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']
                    
                    # for binary features, do nothing
                    # throw error if any nans, infs, or valeus greater than 1e6
                    if np.isnan(trajectories[:, j]).any():
                        raise ValueError(f"NaNs found in normalized feature {feature} in shard {shard_file}")
                    if np.isinf(trajectories[:, j]).any():
                        raise ValueError(f"Infs found in normalized feature {feature} in shard {shard_file}")
                    if np.abs(trajectories[:, j]).max() > 1e6:
                        raise ValueError(f"Extreme values found in normalized feature {feature} in shard {shard_file}")
                      
            
            # split into windows for faster loading later
            seq_buffer = []
            window_start_or_end_idx = [] # for transformer model, we need to know where the last window ends
          # because most trajectory lengths will not be perfectly divisible by window_size
                                        # we need to pad   
            window_ptr = [0]
                                         

            for i, stay_id in enumerate(stay_ids):
                start_idx = index_ptr[i]
                length = index_ptr[i+1] - index_ptr[i]
                end_idx = start_idx + length

                traj = trajectories[start_idx:end_idx]

                # compute number of windows
                n_windows = int(np.ceil(length / self.window_length))

                # compute total length after padding
                total_length = n_windows * self.window_length

                # pad traj if needed (repeat first timestep)    
                pad_length = total_length - length                                      
                
                if config_manager.model_type == "dynamic_deephit_transformer":
                    # for transformer model, pad at the end
                    pad_length = total_length - length
                    # pad with zeros
                    padding = np.tile(np.zeros(traj.shape[1]), (pad_length, 1))
                    traj = np.vstack((traj, padding))
                    window_start_or_end_idx.append(self.window_length  - pad_length - 1)
                else:
                    padding = np.tile(traj[0], (pad_length, 1))
                    traj = np.vstack((padding, traj))
                    window_start_or_end_idx.append(pad_length)  
                
                # Reshape into (n_windows, window_size, n_features)
                windows = traj.reshape(n_windows, self.window_length, traj.shape[1])  

                seq_buffer.append(windows)
                window_ptr.append(window_ptr[-1] + n_windows)
            
            # concatenate all windows into a single array
            seq_array = np.concatenate(seq_buffer, axis=0)
            window_ptr = np.array(window_ptr, dtype=np.int32)
            window_start_or_end_idx = np.array(window_start_or_end_idx, dtype=np.int32)

            if config_manager.model_type == "dynamic_deephit_transformer":
                # index_type
                index_type = 'window_end_idx'
                normalized_shard_file = shard_file.replace("_shard.h5", "_shard_normalized_transformer.h5")
            else:
                index_type = 'first_window_start_idx'
                normalized_shard_file = shard_file.replace("_shard.h5", "_shard_normalized.h5")

            with h5py.File(normalized_shard_file, 'w') as f:
                f.create_dataset('feature_names', data=[name.encode('utf-8') for name in feature_names])
                f.create_dataset('stay_ids', data=stay_ids, dtype=np.int32)
                f.create_dataset('window_ptr', data=window_ptr, dtype=np.int32)
                f.create_dataset(index_type, data=window_start_or_end_idx, dtype=np.int32)
                f.create_dataset(
                    'trajectories', 
                    data=seq_array.astype(np.float32), 
                    chunks=(1, self.window_length, n_features)
                )

            logger.info(f"Saved normalized shard file: {normalized_shard_file}")

        
    def _split_into_dynamic_deephit_dataset(self, dataset, config_manager):
        """
        This function will split the dataframe into the format needed for dynamic deephit

        Note that, in addition to full length windows, we need a number of 
        shorter windows from the trajectory starts. This will allow the model 
        how to predict survival time near the start of the trajectory - this
        could be something like 10 windows with ends randomly sampled from the 
        first window_length timesteps of the trajectories. The rest of the windows
        can be full size - there is no reason to sample partial windows anywhere 
        else in the trajectory (as we will always have full windows at those 
        locations).

        Window indices will determine window END

        It will return a dictionary with the following keys:
        - 
        """
        np.random.seed(42)
        
        self.window_length = config_manager.training_config['window_length']

        # Pre-group the dataframe and convert to dict of arrays (much faster)
        grouped = dataset.dataframe.groupby(self.encounter_col)
        traj_data = {}
        traj_lengths = {}

        for encounter_id, group_df in grouped:
            traj_lengths[encounter_id] = len(group_df)
            # Store as numpy array immediately - avoid repeated DataFrame operations
            traj_data[encounter_id] = {
                'features': group_df[self.input_features].to_numpy(),
                'mortality': group_df['mortality'].iloc[-1]
            }

        # calculate number of windows per trajectory
        # we need to sample some short windows from the start of each trajectory
        # and then more windows depending on trajectory lenght
        # so we will loop throught the traj_lengths and calculate n_windows for each traj
        window_indices = {}
        n_windows = {}
        
        
        # get min gap between time bins
        min_gap = int(np.floor(np.min(np.diff(config_manager.time_bins))))
        if min_gap < 1:
            min_gap = 1

        n_windows_per_bin = int(min_gap/4)
        if n_windows_per_bin < 1:
            n_windows_per_bin = 1

        for encounter_id, traj_length in traj_lengths.items():
            window_ind = []

            time_to_event = np.arange(traj_length, 0, -1)

            for i in range(len(config_manager.time_bins) - 1):
                lower_bound = config_manager.time_bins[i]
                upper_bound = config_manager.time_bins[i + 1]
                                           
                # find all indices where time_to_event is in this bin
                indices_in_bin = np.where((time_to_event <= upper_bound) & 
                                          (time_to_event > lower_bound))[0]
                if len(indices_in_bin) > 0:
                    n_to_sample = min(n_windows_per_bin, len(indices_in_bin))
                    sampled_indices = np.random.choice(indices_in_bin, size=n_to_sample, replace=False).tolist()
                    window_ind += sampled_indices

            window_ind = list(set(window_ind))  # remove duplicates
            window_ind = sorted(window_ind)

            window_indices[encounter_id] = window_ind
            n_windows[encounter_id] = len(window_ind)
              
        # get the total number of windows
        total_windows = sum(n_windows.values())

        # create data storage
        feature_array = np.zeros((total_windows, self.window_length, len(self.input_features)))
        time_to_event_array = np.zeros((total_windows))
        event_array = np.zeros((total_windows))
        mask_array = np.zeros((total_windows, self.window_length))

        # for each encounter_id, pick n_windows start indices randomly as integers between
        # 0 and traj_length
        array_ind = 0
        for encounter_id, window_ind_list in window_indices.items():
            # find start index of traj
            traj_length = traj_lengths[encounter_id]
            features = traj_data[encounter_id]['features']
            event = 2 if traj_data[encounter_id]['mortality'] == 1 else 1

            for i in window_ind_list:
                window_end = i + 1
                window_start = max(0, i + 1 - self.window_length)
                
                # Extract window directly from numpy array
                window_features = features[window_start:window_end]
                window_len = len(window_features)
                
                # Pad using the first column from window_features if needed
                if window_len < self.window_length:
                    pad_width = self.window_length - window_len
                    padding = np.tile(window_features[0], (pad_width, 1))
                    window_features = np.vstack((padding, window_features))
                    mask_array[array_ind, :] = 1
                else:
                    mask_array[array_ind, :] = 1
                
                feature_array[array_ind] = window_features
                time_to_event_array[array_ind] = traj_length - window_end + 1
                event_array[array_ind] = event
                array_ind += 1
    
        time_to_event_array_binned = np.digitize(time_to_event_array, 
                                             config_manager.time_bins, 
                                             right=True) - 1
            
        # create the final dataset
        final_dataset = {
            'features': feature_array,
            'time_to_event': time_to_event_array_binned,
            'event': event_array,
            'mask': mask_array
        }

        return final_dataset


    def _split_into_rl_dataset(self, dataset, config_manager):
        """
        This function will split the dataframe into the format needed for RL
        It will return a dictionary with the following keys:
        - observations
        - actions
        - rewards
        - terminations
        - next_observations
        - next_actions (if running basiq_q (SARSA))
        
        state and action come from a given timestep, reward, next_state and terminated 
        come from the next timestep.
        Therefore, each trajectory will actually be length n_timesteps - 1. So observations
        and actions will have indices 0 to n_timesteps - 2, next_observations and rewards 
        and terminated will have indices 1 to n_timesteps - 1.

        we piece off the rewards and actions and then will have the observations left
        """
        
        ##### SHOULD ADD SOME KIND OF UNIT TEST HERE TO CONFIRM EVERYTHING IS WORKING AS EXPECTED #####

        if self.use_reward:
            rewards = dataset.dataframe[self.reward_col].copy()    
            # process the rewards
            if self.flip_reward:
                rewards = -rewards
            if self.offset_reward != 0.0:
                rewards = rewards + self.offset_reward
            if self.scale_reward != 1.0:
                rewards = rewards * self.scale_reward

            # now compute hourly change in reward (don't need to shift, that is handled below!)
            rewards = rewards.diff().fillna(0)
           
            # clip rewards to avoid extreme outliers; these only get used if not 
            # already set in the config file
            lower_bound_temp = np.percentile(rewards, self.reward_lower_percentile)
            upper_bound_temp = np.percentile(rewards, self.reward_upper_percentile)  

            lower_bound = config_manager.training_config.setdefault('lower_bound', lower_bound_temp)
            upper_bound = config_manager.training_config.setdefault('upper_bound', upper_bound_temp)

            rewards = rewards.clip(lower=lower_bound, upper=upper_bound)

            # make reward positive if specified in config file
            if config_manager.training_config['make_rewards_positive']:
                eps = 1e-6
                rewards = rewards - lower_bound + eps           

        else:
            rewards = pd.Series(0., index=dataset.dataframe.index)

        # mortality needs to be processed as we want to augment transitions with a patient-level label for online analysis
        non_survival_ids = dataset.dataframe[dataset.dataframe['mortality'] == 1][self.encounter_col].unique()  
        # create mortality_df
        mortality_df = dataset.dataframe[[self.encounter_col, 'mortality']]
        # set mortality for all rows for non_survival_ids to 1
        mortality_df.loc[mortality_df[self.encounter_col].isin(non_survival_ids), 'mortality'] = 1
        # create mortality
        mortality = mortality_df['mortality'].copy()

        # Note that terminal rewards must be caculated after we calculate "terminated"
        
        # get the actions and observations
        actions = dataset.dataframe[self.actions].copy()
        observations = dataset.dataframe[self.input_features].copy()

        # compute terminated
        # terminate is 0 everywhere except the final timestep for each patient
        dataset.dataframe['terminated'] = 0
        dataset.dataframe.loc[dataset.dataframe.groupby(self.encounter_col).tail(1).index, 'terminated'] = 1
        terminated = dataset.dataframe['terminated'].copy()
        # drop the terminated column from the dataframe
        dataset.dataframe = dataset.dataframe.drop(columns=['terminated'])

        # process the terminal rewards
        if self.use_terminal_reward:
            terminal_rewards = dataset.dataframe[self.terminal_reward_col].copy()
            if self.flip_terminal_reward:
                terminal_rewards = -terminal_rewards # make mortality = -1
            # make terminal rewards = 1 anywhere terminal_rewards == 0 and 
            # terminated == 1 (i.e. the patient survived)
            terminal_rewards[(terminal_rewards == 0) & (terminated == 1)] = 1
            # offset
            if self.offset_terminal_reward:
                terminal_rewards = terminal_rewards + self.offset_terminal_reward
            # scale
            if self.scale_terminal_reward:
                terminal_rewards = terminal_rewards * self.scale_terminal_reward
            if self.make_terminal_incremental:
                individual_rewards = terminal_rewards[terminal_rewards != 0]
                start_indices = individual_rewards.index + 1
                # concatenate a zeo at the start of start_indices and remove the last index
                start_indices = pd.Index([0]).append(start_indices[:-1])

                for i, reward in enumerate(individual_rewards.items()):
                    start_index = start_indices[i]
                    end_index = reward[0]
                    inc_reward = reward[1] / (end_index - start_index)
                    
                    # clip rewards to -.1 and .1 to avoid extreme outliers
                    inc_reward = np.clip(inc_reward, -0.1, 0.1)
                    
                    # set rewards between start_index and end_index to inc_reward
                    terminal_rewards.loc[start_index:end_index] = inc_reward

        else:
            terminal_rewards = pd.Series(0, index=dataset.dataframe.index)

        # add the terminal rewards to the rewards
        rewards = rewards + terminal_rewards
        
        # get the next observations and, if necessart, next_actions
        next_observations = observations.copy()

        if config_manager.training_config['include_next_actions']:
            next_actions = actions.copy()

        # get the indices of all the trajectory starts, i.e. the first index of each patient
        trajectory_starts = dataset.dataframe.groupby(self.encounter_col).head(1).index

        # get the trajectory ends, i.e. the last index of each patient
        trajectory_ends = dataset.dataframe.groupby(self.encounter_col).tail(1).index

        # for observations and actions, we remove the trajectory ends
        observations = observations.drop(index=trajectory_ends)
        actions = actions.drop(index=trajectory_ends)

        # for mortality, we also remove the trajectory ends (could be the starts, doesn't matter, just need the lenght to match)
        mortality = mortality.drop(index=trajectory_ends)

        # for rewards, next_observations, and terminated, we remove the trajectory starts
        rewards = rewards.drop(index=trajectory_starts)
        next_observations = next_observations.drop(index=trajectory_starts)
        terminated = terminated.drop(index=trajectory_starts)
        if config_manager.training_config['include_next_actions']:
            next_actions = next_actions.drop(index=trajectory_starts)

        # convert to numpy arrays
        observations = observations.to_numpy()  
        actions = actions.to_numpy()
        rewards = rewards.to_numpy()
        next_observations = next_observations.to_numpy()
        if config_manager.training_config['include_next_actions']:
            next_actions = next_actions.to_numpy()
        terminated = terminated.to_numpy()
        mortality = mortality.to_numpy()

        if config_manager.training_config['include_next_actions']:
            return {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_observations,
                "next_actions": next_actions,
                "terminated": terminated,
                "mortality": mortality
            }
        else:
            return {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_observations,
                "terminated": terminated,
                "mortality": mortality
            }
        

    def _create_binary_actions(self, config_manager):
                
        # loop through the columns in self.rl_dataset['actions'] and create a
        # binary action for each action, where 1 means the action was given
        # and 0 means it was not given, adding the column onto the array       
    
        actions_array = self.rl_dataset['actions']
        
        # Vectorized binary creation for all columns at once
        binary_actions = (actions_array > 0).astype(int)
        binary_names = [f"{name}_binary" for name in self.actions]
        original_names = self.actions.copy()
        
        if config_manager.action_mode == 'mixed':
            # Concatenate original and binary actions
            self.rl_dataset['actions'] = np.concatenate([actions_array, binary_actions], axis=1)
            # update action names            
            self.actions = self.actions + binary_names

        elif config_manager.action_mode == 'binary_only':
            self.rl_dataset['actions'] = binary_actions
            # update action names
            self.actions = binary_names

        else:
            raise ValueError(f"Invalid action_mode: {config_manager.action_mode}")
        
        logger.debug(f"Created {binary_actions.shape[1]} binary actions")
        logger.debug(f"Binary actions: {binary_names}")
        logger.debug(f"New action array shape: {self.rl_dataset['actions'].shape}")


    def _normalize_obs(self, dataset):
        # apply min-max normalization to drugs and ordinal features, 
        # leave one hot encoded features as they are, and z-score normalize 
        # the continuous features
        self.physiological_feature_ind = [] # for dynamic deephit model, which tries to predict next physiological features
        self.physiological_feature_type = []

        physiological_feature_sets = ['blood_gas', 'labs', 'neuro', 'vitals']

        if self.obs_norm_dict is None:
            self.obs_norm_dict = {}

        # get the features row in self.features_to_use
        if not hasattr(self, 'features_to_use'):
            # create features_to_use dataframe with self.input_features as 'feature' col,
            # feature_set as 'other', and data_type as 'continuous'
            self.features_to_use = pd.DataFrame({
                'feature': self.input_features,
                'feature_set': ['vitals'] * len(self.input_features),
                'data_type': ['continuous'] * len(self.input_features)
            })        
        
        
        for i, feature in enumerate(self.input_features):

            feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
            data_type = feature_row['data_type'].values[0]
            feature_set = feature_row['feature_set'].values[0]

            if feature_set in physiological_feature_sets:
                self.physiological_feature_ind.append(i)    
                self.physiological_feature_type.append(data_type)        
            
            if feature not in self.obs_norm_dict:
                # self.obs_norm_dict[feature] = {}
                # self.obs_norm_dict[feature]['min'] = self.rl_dataset['observations'][:, i].min()
                # self.obs_norm_dict[feature]['max'] = self.rl_dataset['observations'][:, i].max()
                # self.obs_norm_dict[feature]['mean'] = self.rl_dataset['observations'][:, i].mean()
                # self.obs_norm_dict[feature]['std'] = self.rl_dataset['observations'][:, i].std()

                self.obs_norm_dict[feature] = {}
                self.obs_norm_dict[feature]['min'] = dataset.dataframe[feature].min()
                self.obs_norm_dict[feature]['max'] = dataset.dataframe[feature].max()
                self.obs_norm_dict[feature]['mean'] = dataset.dataframe[feature].mean()
                self.obs_norm_dict[feature]['std'] = dataset.dataframe[feature].std()
                        
            if data_type == 'ordinal' or feature_set == 'infusion' or \
                feature_set == 'other_meds' or feature_set == 'sedation':
                
                # min-max normalize the feature
                normalization_type = 'min-max'
                
                # normalize the ith col in self.rl_dataset['observations']
                # using the min and max from the training set
                max_min_diff = self.obs_norm_dict[feature]['max'] - self.obs_norm_dict[feature]['min']
                if max_min_diff == 0:
                    # self.rl_dataset['observations'][:, i] = 0.0
                    # self.rl_dataset['next_observations'][:, i] = 0.0                
                    dataset.dataframe[feature] = 0.0                
                
                else:
                    # self.rl_dataset['observations'][:, i] = (self.rl_dataset['observations'][:, i] - 
                    #     self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                    #                                         - self.obs_norm_dict[feature]['min'])

                    dataset.dataframe[feature] = (dataset.dataframe[feature] - 
                        self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                                                            - self.obs_norm_dict[feature]['min'])
                    
                    
                    # it is possible here that if both min and max are 0, we will get NaNs;
                    # in this case, we set the normalized value to 0
                    # self.rl_dataset['observations'][:, i] = \
                    #     np.nan_to_num(self.rl_dataset['observations'][:, i], nan=0.0)
                    
                    dataset.dataframe[feature] = \
                        np.nan_to_num(dataset.dataframe[feature], nan=0.0)
                
                    # it is possible to get very large values that are less than inf, set them to zero
                    # self.rl_dataset['observations'][:, i] = \
                    #     np.where(np.abs(self.rl_dataset['observations'][:, i]) > 1e6, 0.0, self.rl_dataset['observations'][:, i])                
                    
                    dataset.dataframe[feature] = \
                        np.where(np.abs(dataset.dataframe[feature]) > 1e6, 0.0, dataset.dataframe[feature])
                    
                    
                    # also possible to get Infs if max == min but numerator is non-zero, so we set those to 0 as well
                    # self.rl_dataset['observations'][:, i] = \
                    #     np.where(np.isinf(self.rl_dataset['observations'][:, i]), 0.0, self.rl_dataset['observations'][:, i])
                    
                    dataset.dataframe[feature] = \
                        np.where(np.isinf(dataset.dataframe[feature]), 0.0, dataset.dataframe[feature])
                    
                    # also normalize the next_observations - NO LONGER NECESSRY AS WE ARE NORMALIZING THE DATAFRAME DIRECTLY
                    # self.rl_dataset['next_observations'][:, i] = (self.rl_dataset['next_observations'][:, i] - 
                    #     self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                    #                                         - self.obs_norm_dict[feature]['min'])                       

                    # self.rl_dataset['next_observations'][:, i] = \
                    #     np.where(np.abs(self.rl_dataset['next_observations'][:, i]) > 1e6, 0.0, self.rl_dataset['next_observations'][:, i])

                    # self.rl_dataset['next_observations'][:, i] = \
                    #     np.nan_to_num(self.rl_dataset['next_observations'][:, i], nan=0.0)   

                    # self.rl_dataset['next_observations'][:, i] = \
                    #     np.where(np.isinf(self.rl_dataset['next_observations'][:, i]), 0.0, self.rl_dataset['next_observations'][:, i])  

            elif data_type == 'binary':
                # binary features are already normalized, so we just store the normalization type
                normalization_type = 'none'

            elif data_type == 'continuous':
                # z-score normalize the feature
                normalization_type = 'z-score'

                # normalize the ith col in self.rl_dataset['observations']
                # using the mean and std from the training set
                # self.rl_dataset['observations'][:, i] = (self.rl_dataset['observations'][:, i] - 
                #     self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']
                
                dataset.dataframe[feature] = (dataset.dataframe[feature] - 
                    self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']
                
                # also normalize the next_observations
                # self.rl_dataset['next_observations'][:, i] = (self.rl_dataset['next_observations'][:, i] - 
                #     self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']

            # check for NaNs in the normalized feature
            # if np.isnan(self.rl_dataset['observations'][:, i]).any() or \
            #    np.isnan(self.rl_dataset['next_observations'][:, i]).any():
            #     raise ValueError(f"NaNs found in normalized feature {feature}")

            # check for NaNs in the normalized feature
            if np.isnan(dataset.dataframe[feature]).any() :
                raise ValueError(f"NaNs found in normalized feature {feature}")

            # check for Infs in the normalized feature
            # if np.isinf(self.rl_dataset['observations'][:, i]).any() or \
            #    np.isinf(self.rl_dataset['next_observations'][:, i]).any():
            #     raise ValueError(f"Infs found in normalized feature {feature}") 
            
            if np.isinf(dataset.dataframe[feature]).any():
                raise ValueError(f"Infs found in normalized feature {feature}") 


            logger.debug(f"Normalized feature {feature} with {normalization_type} normalization. "
                f"Min: {self.obs_norm_dict[feature]['min']}, \
                    Max: {self.obs_norm_dict[feature]['max']}, \
                        Mean: {self.obs_norm_dict[feature]['mean']}, \
                            Std: {self.obs_norm_dict[feature]['std']}")
            
            if self.obs_norm_dict[feature]['max'] > 5:
                pass

            if self.obs_norm_dict[feature]['min'] < -5:
                pass

        logger.debug("Observation normalization complete.")                    

    
    def _normalize_actions(self):
        # normalize actions between 0 and 1
        if self.action_minimum is None and self.action_maximum is None:
            self.action_minimum = {}
            self.action_maximum = {}
        
        for i, action in enumerate(self.actions):
            # normalize the ith col in self.rl_dataset['actions']
            # using min-max normalization between 0 and 1;
            if action not in self.action_minimum and action not in self.action_maximum:
                self.action_minimum[action] = self.rl_dataset['actions'][:, i].min()
                self.action_maximum[action] = self.rl_dataset['actions'][:, i].max()

            self.rl_dataset['actions'][:, i] = (self.rl_dataset['actions'][:, i] 
                - self.action_minimum[action]) / (self.action_maximum[action] - self.action_minimum[action])
            
            # if next_actions exist, normalize them too
            if 'next_actions' in self.rl_dataset:
                self.rl_dataset['next_actions'][:, i] = (self.rl_dataset['next_actions'][:, i] 
                    - self.action_minimum[action]) / (self.action_maximum[action] - self.action_minimum[action])

            logger.debug(f"Normalized action {action} between 0 and 1.")        


    def _transform_actions(self):

        # Store the fitted transformers for later use
        if self.action_transformers is None:
            self.action_transformers = {}

        for i, target in enumerate(self.actions):
            target_transform = self.action_transforms[i]

            if target_transform == "yeo-johnson": # no other transforms yet implemented
                
                if target not in self.action_transformers:
                    # fit the transformer on the training data
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)    
                    pt.fit(self.rl_dataset['actions'][:, i].reshape(-1, 1))  # Reshape to 2D array
                    
                    # fit only on the non-zero values to avoid changing the meaning of zero
                    non_zero_indices = self.rl_dataset['actions'][:, i] != 0
                    pt.fit(self.rl_dataset['actions'][non_zero_indices, i].reshape(-1, 1))
                    
                    self.action_transformers[target] = pt  

                else:
                    pt = self.action_transformers[target]            

                # transform the training data
                self.rl_dataset['actions'][:, i] = pt.transform(
                    self.rl_dataset['actions'][:, i].reshape(-1, 1)).flatten()

                # if next_actions exist, transform them too
                if 'next_actions' in self.rl_dataset:
                    self.rl_dataset['next_actions'][:, i] = pt.transform(
                        self.rl_dataset['next_actions'][:, i].reshape(-1, 1)).flatten()
                                     
                logger.debug(f"Transformed target {target} with Yeo-Johnson transform.")

            elif target_transform == "log1p":
                # apply log1p transform, which is log(1 + x)
                # this will leave zeros as zeros, and compress the higher values
                self.rl_dataset['actions'][:, i] = np.log1p(self.rl_dataset['actions'][:, i])
                
                # if next_actions exist, transform them too
                if 'next_actions' in self.rl_dataset:
                    self.rl_dataset['next_actions'][:, i] = np.log1p(self.rl_dataset['next_actions'][:, i])

                logger.debug(f"Transformed target {target} with log1p transform.")

    
    def get_normalization_params(self):

        # check if self has the "obs_norm_dict" attribute
        if not hasattr(self, 'obs_norm_dict'):
            return None

        return {
            "obs_norm_dict": self.obs_norm_dict,
            "action_minimum": self.action_minimum,
            "action_maximum": self.action_maximum,
            "action_transformers": self.action_transformers
        }
                
        
def main():
    pass

if __name__ == "__main__":
    main()