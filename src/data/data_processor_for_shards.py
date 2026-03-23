"""
Creates DataProcessor class to process data stored in HDF5 format.
All normalization and transformation is done here.
"""
import yaml
import copy
import h5py
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import json
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler

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

        if self.split == 'train' and config_manager.training_config['tuning']:
            self.split = 'tuning'
        
        # find the shard dir for the given split
        if config_manager.split_num is None:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'])
        else:
            shard_dir = os.path.join(config_manager.paths['data_dir'], 
                        config_manager.data_config['shard_dir'], 
                        f"split_{config_manager.split_num}")

        # switch to tuning dir if we're in tuning mode
        if not config_manager.training_config['tuning']:
            self.shard_dir = os.path.join(shard_dir, 'training_set')
        else:
            self.shard_dir = os.path.join(shard_dir, 'tuning_set')
        
        
        # load processed data if it exists, otherwise process raw shard files and save the processed data for future use
        final_file_name = f"{self.split}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)
        
        # load if exists
        if os.path.exists(final_file_path):
            logger.info(f"Loading existing dynamic deephit dataset from {final_file_path}")
            
            # load the file directly into self.dynamic_deephit_dataset
            self.dynamic_deephit_dataset = self.load_dynamic_deephit_dataset_from_h5(final_file_path)
            
            # if we are running RL, load reward
            if config_manager.data_config['model_type'] == 'reinforcement_learning' and config_manager.data_config['use_reward']:
                # get the parent directory for the supervised model which generates the health score
                supervised_model_dir = os.path.dirname(config_manager.paths['supervised_final_model_dir'])
                supervised_model_dir = os.path.join(supervised_model_dir, "five_splits", 
                                                       f"split_{config_manager.split_num}", "final_model")
                # get the folder inside
                final_model_dir = os.listdir(supervised_model_dir)[0]

                pred_split = self.split
                if self.split == 'tuning':
                    pred_split = 'train' # the tuning set predictions are generated from the model trained on the training set, so we need to load the train_set_predictions.h5 file
                prediction_file_path = os.path.join(supervised_model_dir, final_model_dir, f"{pred_split}_set_predictions.h5")

                self.dynamic_deephit_predictions = self.load_dynamic_deephit_predictions_from_h5(prediction_file_path, 
                                                                        keys_to_exclude=['predictions', 'entropy'])
                
                if self.split == 'tuning': # we need to filter the predictions to only include the stays in the tuning set, which are a subset of the training set
                    tuning_stay_ids = self.dynamic_deephit_dataset['stay_ids']
                    self.filter_by_stay_ids(tuning_stay_ids, dataset_name='dynamic_deephit_predictions')

                # process the reward
                self.process_reward(config_manager)

                return

            
            # load the normalization parameters if they exist from the json file
            if config_manager.training_config['tuning']:
                norm_params_file = os.path.join(self.shard_dir, "tuning_norm_params.json")
            else:
                norm_params_file = os.path.join(self.shard_dir, "train_norm_params.json")
            if os.path.exists(norm_params_file):
                with open(norm_params_file, 'r') as f:
                    norm_params = json.load(f)
                self.obs_norm_dict = norm_params['obs_norm_dict']
                logger.info(f"Loaded normalization parameters from {norm_params_file}")
            else:
                raise ValueError(f"No normalization parameters file found at {norm_params_file}. This may cause issues if you are normalizing the validation or test set, as it should be normalized with the parameters from the training set. Please check that your training set was normalized and that the normalization parameters were saved correctly.")

            return
           
        
        # prepare for normalization
        self.norm_params = norm_params
        self.window_length = config_manager.training_config['window_length']
        self.time_bins = config_manager.time_bins

        self.max_time = self.time_bins[-1]

        # load features to use from config_manager
        self.features_to_use = pd.read_csv(config_manager.data_config['features_to_use_path'])       

        all_files = os.listdir(self.shard_dir)                

        raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith(".h5")]
        if len(raw_shard_files) == 0:
            raise ValueError(f"No shard files found for split {self.split} in directory {self.shard_dir}")
        elif len(raw_shard_files) > 1:
            # I've simplified the code to work from only one file - check v1 and v2 to get an idea how to use multiple shards, but didn't implement robust scaling yet, which requires median and IQR, which is trickier for multiple shards
            raise ValueError(f"Multiple shard files found for split {self.split} in directory {self.shard_dir}. Expected only one. Found: {raw_shard_files}")

        # don't need a list, as there should only be one file, confirmed above
        self.raw_shard_file = raw_shard_files[0]
            
        ########### Normalized Shards ##############
        self.normalize(config_manager=config_manager) # this creates self.dynamic_deephit_dataset
            
        # create the time_to_event array
        time_to_event = np.zeros_like(self.dynamic_deephit_dataset['trajectories'][:,0], dtype=np.int32)
        for j in range(len(self.dynamic_deephit_dataset['stay_ids'])):
            start_idx = self.dynamic_deephit_dataset['index_ptr'][j]
            end_idx = self.dynamic_deephit_dataset['index_ptr'][j+1]
            length = end_idx - start_idx
            tte = np.arange(length, 0, -1)
            time_to_event[start_idx:end_idx] = tte

        self.dynamic_deephit_dataset['time_to_event_raw'] = time_to_event.astype(np.int32)

        # bin time_to_events
        tte_bin = np.digitize(self.dynamic_deephit_dataset['time_to_event_raw'], self.time_bins, right=True) - 1                
        self.dynamic_deephit_dataset['time_to_event'] = tte_bin

        # get physiological feature indices for dynamic deephit model, and feature_dim
        self.dynamic_deephit_dataset['feature_dim'] = self.dynamic_deephit_dataset['trajectories'].shape[1]

        self.dynamic_deephit_dataset['physiological_feature_ind'] = [] # for dynamic deephit model, which tries to predict next physiological features
        self.dynamic_deephit_dataset['physiological_feature_type'] = []
        self.dynamic_deephit_dataset['response_zero_ind'] = [] # for final timestep features that should be zeroed out because they are responses to interventions at that timestep, and we don't want to feed the model the true response values at the final timestep, as that would leak information about the target
        self.dynamic_deephit_dataset['carry_forward_ind'] = [] # for rinal timestep features that should be carry-forwarded because they are responses to interventions at that timestep, and we want to carry forward the last observed value instead of using the true response value at the final timestep, to avoid leaking information about the target, but also to provide a more realistic scenario for the model to learn from, as in real life, we wouldn't have access to the true response values at the final timestep during inference, but we would have access to the last observed values that we can carry forward.
        
        physiological_feature_sets = [
            'hemodynamics', 'labs', 'neurological', 'routine_vital_signs', 'respiratory'
            ]
                
        for i, feature in enumerate(self.dynamic_deephit_dataset['feature_names']):
            if feature == 'time':
                continue
            
            if feature in self.features_to_use['feature'].values:
                feature_row = self.features_to_use[self.features_to_use['feature'] == feature]            
                feature_set = feature_row['feature_set'].values[0]
                data_type = feature_row['data_type'].values[0]
                response_intervention = feature_row['response_intervention'].values[0]
                
                if feature_set in physiological_feature_sets and response_intervention == 'response':
                    self.dynamic_deephit_dataset['physiological_feature_ind'].append(i)
                    self.dynamic_deephit_dataset['physiological_feature_type'].append(data_type)

                elif response_intervention == 'intervention':
                    if feature_set in ['fluids_intake', 'medications']:
                        self.dynamic_deephit_dataset['response_zero_ind'].append(i)
                    else:
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(i)
                
                    if f"{feature}_mask" in self.dynamic_deephit_dataset['feature_names']:
                        mask_ind = self.dynamic_deephit_dataset['feature_names'].index(f"{feature}_mask")
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(mask_ind)
                    
                    if f"{feature}_delta" in self.dynamic_deephit_dataset['feature_names']:
                        delta_ind = self.dynamic_deephit_dataset['feature_names'].index(f"{feature}_delta")
                        self.dynamic_deephit_dataset['carry_forward_ind'].append(delta_ind)

        
        # save the file
        final_file_name = f"{self.split}_dynamic_deephit_dataset.h5"
        final_file_path = os.path.join(self.shard_dir, final_file_name)

        # save the dynamic deephit dataset to h5 file for future use
        self.feature_dim = self.dynamic_deephit_dataset['trajectories'].shape[1]
        self.dynamic_deephit_dataset_to_h5(final_file_path)
        
        if config_manager._data_config['cohort'] is not None and config_manager._data_config['cohort'] != 'all':
            # filter the dataset to only include the specified cohort
            logger.info(f"Filtering dynamic deephit dataset to only include cohort: {config_manager._data_config['cohort']}")
            self.filter_cohort(config_manager)

    
    def process_reward(self, config_manager):
        reward_type = config_manager.data_config['reward_type']
        if reward_type == 'health_score':
            health_score = self.dynamic_deephit_predictions['health_score']
            
            # health score is from 0 to 100, we need to centre it so it is -50 to 50
            health_score = health_score - 50.0            
            
            index_ptr = self.dynamic_deephit_dataset['index_ptr']

            # calculate hourly change; first hour should be 0
            hourly_health_change = np.zeros_like(health_score)
            for j in range(len(index_ptr) - 1):
                start_idx = index_ptr[j]
                end_idx = index_ptr[j + 1]
                hourly_health_change[start_idx:end_idx] = np.diff(health_score[start_idx:end_idx], append=health_score[end_idx-1])

            # reward_std = hourly_health_change.std()
            #clip rewards to -5, 5
            hourly_health_change = np.clip(hourly_health_change, -5.0, 5.0)
           
            self.dynamic_deephit_predictions['reward'] = hourly_health_change.astype(np.float32)                                         
        
        else:
            raise ValueError(f"Invalid reward type: {reward_type}. Supported reward types are: 'health_score' and 'time_to_event'.")
    
    


    def create_rl_dataset(self, states="model_outputs", scaler=None):
        self.rl_dataset = {}
        
        action_ind = self.dynamic_deephit_dataset['response_zero_ind'] + self.dynamic_deephit_dataset['carry_forward_ind']
        action_ind.sort()  # these are the indices of the response features that are either zeroed out or carried forward at the final timestep, so they correspond to the action space

        action_names = np.array(self.dynamic_deephit_dataset['feature_names'])[action_ind]

        # find all actions with 'delta' or 'mask' in the name and remove them from the action space (since these are not actual treatments but rather engineered features for the model)
        remove_mask = np.array([('delta' in name) or ('mask' in name) for name in action_names])

        action_ind = np.array(action_ind)[~remove_mask]
        self.action_names = action_names[~remove_mask]

        # get trajectory starts and ends
        all_ind = np.arange(len(self.dynamic_deephit_dataset['trajectories']))
        trajectory_start_inds = self.dynamic_deephit_dataset['index_ptr'][:-1]
        trajectory_end_inds = self.dynamic_deephit_dataset['index_ptr'][1:] - 1

        state_ind = np.setdiff1d(all_ind, trajectory_end_inds)  # we will exclude the end of trajectory timesteps since these have no next state and action labels are ambiguous (since they could be either the action taken at that timestep or the action taken at the previous timestep)
        next_state_ind = np.setdiff1d(all_ind, trajectory_start_inds)  # we will exclude the start of trajectory timesteps from the next states for the same reason as above

        # get states
        if states == "model_outputs":
            # need to scale the outputs
            if scaler == None:
                # use StandardScaler to scale the model outputs, fit on the states in the training set
                scaler = StandardScaler()
                scaler.fit(self.dynamic_deephit_predictions["model_outputs"])
            
            # apply the scaler to the model outputs
            scaled_model_outputs = scaler.transform(self.dynamic_deephit_predictions["model_outputs"])                            
            
            self.rl_dataset['states'] = torch.tensor(scaled_model_outputs[state_ind, :], dtype=torch.float32)
            self.rl_dataset['next_states'] = torch.tensor(scaled_model_outputs[next_state_ind, :], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Only states='model_outputs' is currently implemented, but got states={states}")

        actions = torch.tensor(self.dynamic_deephit_dataset["trajectories"][:, action_ind], dtype=torch.float32)
        self.rl_dataset['actions'] = actions[state_ind, :]

        self.rl_dataset['rewards'] = torch.tensor(self.dynamic_deephit_predictions["reward"][state_ind], dtype=torch.float32)
      
        self.rl_dataset['next_actions'] = actions[next_state_ind, :]
        
        dones = torch.tensor(np.zeros_like(self.dynamic_deephit_predictions["reward"]), dtype=torch.float32)
        dones[trajectory_end_inds] = 1.0  # mark end of trajectories
        self.rl_dataset['dones'] = dones[next_state_ind]  # only keep dones for the next states, since the current states will never be terminal in our setting (since we exclude end of trajectory timesteps)
        
        mortality = self.dynamic_deephit_dataset['event']
        mortality_full = np.zeros_like(self.dynamic_deephit_predictions["reward"], dtype=np.float32)
        # make full mortality trajectoires == 1
        mortality_start_inds = trajectory_start_inds[mortality == 1]
        mortality_end_inds = trajectory_end_inds[mortality == 1]

        for start, end in zip(mortality_start_inds, mortality_end_inds):
            mortality_full[start:end+1] = 1.0

        mortality_full[trajectory_end_inds] = mortality  # set the mortality value at the start of each trajectory to the mortality value for that trajectory
        mortality_full = mortality_full[next_state_ind]  # only keep mortality for the next states, since the current states will never be terminal in our setting (since we exclude end of trajectory timesteps)
        self.rl_dataset['mortality'] = torch.tensor(mortality_full, dtype=torch.float32)

        return scaler


    def get_start_states(self):
        dones = self.rl_dataset['dones'].numpy()
        final_ind = np.where(dones == 1)[0]
        start_ind = np.concatenate([[0], final_ind[:-1] + 1])  # start indices are just after each final index, except for the first one which is 0
        return self.rl_dataset['states'][start_ind]



    def dynamic_deephit_dataset_to_h5(self, file_path, keys_to_exclude=None):
        if keys_to_exclude is None:
            keys_to_exclude = []
        
        with h5py.File(file_path, 'w') as f:
            # any dataset should have at minimum feature_names, stay_ids, index_ptr, 
            # f.create_dataset('feature_names', data=self.dynamic_deephit_dataset['feature_names'])
            f.create_dataset('feature_names', data=np.array([name.encode('utf-8') for name in self.dynamic_deephit_dataset['feature_names']], dtype='S'))
            f.create_dataset('stay_ids', data=self.dynamic_deephit_dataset['stay_ids'], dtype=np.int32)
            f.create_dataset('index_ptr', data=self.dynamic_deephit_dataset['index_ptr'], dtype=np.int32)
            f.create_dataset('response_zero_ind', data=np.array(self.dynamic_deephit_dataset['response_zero_ind'], dtype=np.int32), dtype=np.int32)
            f.create_dataset('carry_forward_ind', data=np.array(self.dynamic_deephit_dataset['carry_forward_ind'], dtype=np.int32), dtype=np.int32)
            
            if 'trajectories' in self.dynamic_deephit_dataset and 'trajectories' not in keys_to_exclude:
                f.create_dataset('trajectories', data=self.dynamic_deephit_dataset['trajectories'], dtype=np.float32, 
                                chunks=(1, self.dynamic_deephit_dataset['trajectories'].shape[1]))
            if 'event' in self.dynamic_deephit_dataset and 'event' not in keys_to_exclude:            
                f.create_dataset('event', data=self.dynamic_deephit_dataset['event'], dtype=np.int32)
            
            if 'time_to_event_raw' in self.dynamic_deephit_dataset and 'time_to_event_raw' not in keys_to_exclude:
                f.create_dataset('time_to_event_raw', data=self.dynamic_deephit_dataset['time_to_event_raw'], dtype=np.int32)
                
            if 'time_to_event' in self.dynamic_deephit_dataset and 'time_to_event' not in keys_to_exclude:
                f.create_dataset('time_to_event', data=self.dynamic_deephit_dataset['time_to_event'], dtype=np.int32)
            
            if 'physiological_feature_ind' in self.dynamic_deephit_dataset and 'physiological_feature_ind' not in keys_to_exclude:
                f.create_dataset('physiological_feature_ind', data=np.array(self.dynamic_deephit_dataset['physiological_feature_ind'], dtype=np.int32), dtype=np.int32)
            if 'physiological_feature_type' in self.dynamic_deephit_dataset and 'physiological_feature_type' not in keys_to_exclude:
                f.create_dataset('physiological_feature_type', data=np.array(self.dynamic_deephit_dataset['physiological_feature_type'], dtype='S20'), dtype='S20')
            if 'feature_dim' in self.dynamic_deephit_dataset and 'feature_dim' not in keys_to_exclude:
                f.create_dataset('feature_dim', data=np.array(self.dynamic_deephit_dataset['feature_dim'], dtype=np.int32), dtype=np.int32)
            
            if 'predictions' in self.dynamic_deephit_dataset:
                f.create_dataset('predictions', data=self.dynamic_deephit_dataset['predictions'], dtype=np.float32)
            if 'entropy' in self.dynamic_deephit_dataset:
                f.create_dataset('entropy', data=self.dynamic_deephit_dataset['entropy'], dtype=np.float32)
            if 'health_score' in self.dynamic_deephit_dataset:
                f.create_dataset('health_score', data=self.dynamic_deephit_dataset['health_score'], dtype=np.float32)
            if 'model_outputs' in self.dynamic_deephit_dataset:
                f.create_dataset('model_outputs', data=self.dynamic_deephit_dataset['model_outputs'], dtype=np.float32)


        logger.info(f"Saved dynamic deephit dataset to {file_path}")


    
    
    
    def filter_by_stay_ids(self, stay_ids_to_keep, dataset_name='dynamic_deephit_dataset'):
        # filter the dynamic_deephit_dataset
        if dataset_name == 'dynamic_deephit_dataset':
            dataset = self.dynamic_deephit_dataset
        elif dataset_name == 'dynamic_deephit_predictions':
            dataset = self.dynamic_deephit_predictions
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}. Supported dataset names are: 'dynamic_deephit_dataset' and 'dynamic_deephit_predictions'.")

        stay_ids = dataset['stay_ids']
        logger.info(f"Original number of stays in {dataset_name}: {len(stay_ids)}")
        
        # Create boolean mask for stays to keep
        mask = np.isin(stay_ids, list(stay_ids_to_keep))
        indices_to_keep = np.where(mask)[0]
        
        # logger.info(f"Number of stays after filtering to cohort {cohort}: {len(indices_to_keep)}")
        logger.info(f"Number of stays after filtering by stay ids: {len(indices_to_keep)}")

        # Build trajectory mask
        if 'trajectories' in dataset:
            traj_mask = np.zeros(dataset['trajectories'].shape[0], dtype=bool)
        elif 'model_outputs' in dataset:
            traj_mask = np.zeros(dataset['model_outputs'].shape[0], dtype=bool)
        logger.info(f"Original number of timesteps in {dataset_name}: {traj_mask.shape[0]}")

        for idx in indices_to_keep:
            start_idx = dataset['index_ptr'][idx]
            end_idx = dataset['index_ptr'][idx + 1]
            traj_mask[start_idx:end_idx] = True

        # Apply masks to all arrays at once
        dataset['stay_ids'] = dataset['stay_ids'][mask]
        if 'event' in dataset:
            dataset['event'] = dataset['event'][mask]
        if 'trajectories' in dataset:
            dataset['trajectories'] = dataset['trajectories'][traj_mask]
        if 'time_to_event_raw' in dataset:
            dataset['time_to_event_raw'] = dataset['time_to_event_raw'][traj_mask]
        if 'time_to_event' in dataset:
            dataset['time_to_event'] = dataset['time_to_event'][traj_mask]
        if 'model_outputs' in dataset:
            dataset['model_outputs'] = dataset['model_outputs'][traj_mask]
        if 'health_score' in dataset:
            dataset['health_score'] = dataset['health_score'][traj_mask]
        
        # Rebuild index_ptr
        new_index_ptr = np.zeros(len(indices_to_keep) + 1, dtype=np.int32)
        for i, idx in enumerate(indices_to_keep):
            start_idx = dataset['index_ptr'][idx]
            end_idx = dataset['index_ptr'][idx + 1]
            new_index_ptr[i + 1] = new_index_ptr[i] + (end_idx - start_idx)
        
        dataset['index_ptr'] = new_index_ptr
        # logger.info(f"Number of timesteps after filtering to cohort {cohort}: {dataset['trajectories'].shape[0]}")
        if 'trajectories' in dataset:
            logger.info(f"Number of timesteps after filtering by stay ids: {dataset['trajectories'].shape[0]}")
        elif 'model_outputs' in dataset:
            logger.info(f"Number of timesteps after filtering by stay ids: {dataset['model_outputs'].shape[0]}")

        # Explicitly update the instance attribute
        if dataset_name == 'dynamic_deephit_dataset':
            self.dynamic_deephit_dataset = dataset
        elif dataset_name == 'dynamic_deephit_predictions':
            self.dynamic_deephit_predictions = dataset

        return


    def filter_cohort(self, config_manager):
        """
        Filters the dynamic_deephit_dataset to only include the specified cohort.
        """
        cohort = config_manager._data_config['cohort']
        if cohort is None or cohort == 'all':
            return
        
        # load cohort stay_ids from csv
        stay_ids_path = config_manager.data_config['cohort_stay_ids_path']
        cohort_df = pd.read_csv(stay_ids_path)
        cohort_stay_ids = set(cohort_df['stay_id'].unique().tolist())
        # sort cohort_stay_ids
        cohort_stay_ids = sorted(cohort_stay_ids)
        logger.info(f"Number of stays in cohort {cohort}: {len(cohort_stay_ids)}")
        logger.info(f"Filtering dynamic_deephit_dataset to only include these stay_ids.")

        self.filter_by_stay_ids(cohort_stay_ids, config_manager)    
    
    
    def add_to_dynamic_deephit_dataset_h5(self, file_path, key):
        # get dimensionality of data to be saved
        data = self.dynamic_deephit_dataset[key]
        dimensions = data.shape
        chunks = (1, ) + dimensions[1:]
                
        with h5py.File(file_path, 'a') as f:
            f.create_dataset(key, data=self.dynamic_deephit_dataset[key], dtype=np.float32, 
                chunks=chunks)
            

    def drop_long_trajectories(self, trajectories, stay_ids, index_ptr):
        """
        Drops trajectories longer than max_time from the given trajectories, stay_ids, and index_ptr.
        Returns the updated trajectories, stay_ids, and index_ptr.
        """
        trajectory_lengths = index_ptr[1:] - index_ptr[:-1]

        trajectories_to_remove = np.where(trajectory_lengths > self.max_time)[0]
        stay_ids_to_remove = stay_ids[trajectories_to_remove]
                
        for stay_id in stay_ids_to_remove:
            idx = np.where(stay_ids == stay_id)[0][0]
            start_idx = index_ptr[idx]
            end_idx = index_ptr[idx + 1]

            trajectories = np.delete(trajectories, np.s_[start_idx:end_idx], axis=0)
            stay_ids = np.delete(stay_ids, idx, axis=0)
            index_ptr = np.delete(index_ptr, idx + 1, axis=0)
            # update index_ptr
            index_ptr[idx + 1:] -= (end_idx - start_idx)
        
        return trajectories, stay_ids, index_ptr

    
    def add_time_column(self, trajectories, stay_ids, index_ptr, feature_names):
        n_samples = trajectories.shape[0]
        time_col = np.zeros((n_samples, 1))
        for j in range(len(stay_ids)):
            start_idx = index_ptr[j]
            end_idx = index_ptr[j+1]
            length = end_idx - start_idx
            time_values = np.arange(1, length + 1).reshape(-1, 1)
            time_col[start_idx:end_idx, 0] = time_values[:, 0]
        # append time_col to trajectories
        trajectories = np.hstack((time_col, trajectories))
        return trajectories, ['time'] + feature_names

    
    def min_max(self, x, norm_dict):
        min_val = norm_dict['min']
        max_val = norm_dict['max']
        if max_val - min_val == 0:
            return np.zeros_like(x)
        scaled_x = (x - min_val) / (max_val - min_val)
        return np.clip(scaled_x, 0.0, 1.0)


    def log_min_max(self, x, norm_dict):
        log_x = np.log1p(x)
        
        log_min = norm_dict['log_min']
        log_max = norm_dict['log_max']
        max_min_diff = log_max - log_min
        
        if max_min_diff == 0:
            return np.zeros_like(log_x)
        
        scaled_x = (log_x - log_min) / max_min_diff

        # clip to [0, 1] to handle OOD in val/test
        return np.clip(scaled_x, 0.0, 1.0)


    def z_score(self, x, norm_dict):
        mean = norm_dict['mean']
        std = norm_dict['std']
        if std == 0:
            return np.zeros_like(x)

        # clip to [5,-5] to handle extreme outliers
        scaled_x = (x - mean) / std
        return np.clip(scaled_x, -5.0, 5.0)

    
    def robust_scale(self, x, norm_dict):
        median = norm_dict['median']
        iqr = norm_dict['iqr']
        if iqr == 0:
            return np.zeros_like(x)

        scaled_x = (x - median) / iqr
        # clip to [-5, 5] to handle extreme outliers
        return np.clip(scaled_x, -5.0, 5.0)


    def normalize(self, config_manager=config_manager):
        # is norm_params already exists, it's because it was generated from the training or tuning set and we're now normalizing val or test
        if self.norm_params is not None:
            self.obs_norm_dict = self.norm_params['obs_norm_dict']

        # if not, we are normalizing the training or tuning set, so we need to compute the normalization parameters from the data
        else:
            self.obs_norm_dict = {}

            logger.info(f"Computing values for normalizing shard file: {self.raw_shard_file}")
            # load the shard file
            with h5py.File(self.raw_shard_file, 'r') as f:      
                # convert feature_names from bytes to strings
                feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
                stay_ids = f['stay_ids'][:]
                index_ptr = f['index_ptr'][:]
                trajectories = f['trajectories'][:]             
                
            # drop trajectories longer than max_time
            # remove long trajectories beyond max_time
            trajectories, stay_ids, index_ptr = self.drop_long_trajectories(
                trajectories, stay_ids, index_ptr)
            
            if config_manager.data_config['add_time_col']:
                # add time column to trajectories
                trajectories, feature_names = self.add_time_column(trajectories, stay_ids, index_ptr, feature_names)
            
            # get the index of the death feature, and drop it from the trajectories
            death_feature_index = feature_names.index('death')
            # update feature_names
            feature_names.pop(death_feature_index)
            trajectories = np.delete(trajectories, death_feature_index, axis=1)
            
            # compute sums, squared sums, counts, minima, maxima for 1 feature at a time to avoid floating point catastophic cancellation
            for j, feature in enumerate(feature_names):
                self.obs_norm_dict[feature] = {}

                feature_trajectories = trajectories[:, j]
                
                self.obs_norm_dict[feature]['min'] = np.min(feature_trajectories)
                self.obs_norm_dict[feature]['max'] = np.max(feature_trajectories)
                self.obs_norm_dict[feature]['mean'] = np.mean(feature_trajectories)
                self.obs_norm_dict[feature]['std'] = np.std(feature_trajectories)
                self.obs_norm_dict[feature]['median'] = np.median(feature_trajectories)
                self.obs_norm_dict[feature]['iqr'] = np.percentile(feature_trajectories, 75) - np.percentile(feature_trajectories, 25)
                self.obs_norm_dict[feature]['log_min'] = np.min(np.log1p(feature_trajectories))
                self.obs_norm_dict[feature]['log_max'] = np.max(np.log1p(feature_trajectories))

            # save the normalization parameters to a json file for future use when normalizing val and test sets
            norm_params_file = os.path.join(self.shard_dir, f"{self.split}_norm_params.json")
            with open(norm_params_file, 'w') as f:
                json.dump({'obs_norm_dict': self.obs_norm_dict}, f)
            logger.info(f"Saved normalization parameters to {norm_params_file}")
        
        
        
        # now normalize the features with either the pre-computed normalization parameters from the training/tuning set, or the ones we just computed if this is the training/tuning set    
        logger.info(f"Normalizing shard file: {self.raw_shard_file}")
        # load the shard file
        with h5py.File(self.raw_shard_file, 'r') as f:      
            # convert feature_names from bytes to strings
            feature_names = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]                
            stay_ids = f['stay_ids'][:]
            index_ptr = f['index_ptr'][:]
            trajectories = f['trajectories'][:]

        # drop trajectories longer than max_time
        # remove long trajectories beyond max_time
        trajectories, stay_ids, index_ptr = self.drop_long_trajectories(
            trajectories, stay_ids, index_ptr)
        
        if config_manager.data_config['add_time_col']:
            # add time column to trajectories
            trajectories, feature_names = self.add_time_column(trajectories, stay_ids, index_ptr, feature_names)        

        # get the index of the death feature
        death_feature_index = feature_names.index('death')
        
        # extract death feature values before removing
        death_values = trajectories[:, death_feature_index]
        
        # drop death and update feature_names
        feature_names.pop(death_feature_index)
        trajectories = np.delete(trajectories, death_feature_index, axis=1)

        # normalize the trajectories
        # Note: the normalizatio_type column of features_to_use.csv specifies the type of normalization
        # to apply to each feature, except for time, which we have just created, and the delta and mask features,
        # which are normalized in a consistent way
        for j, feature in enumerate(feature_names):   
            # TIME - log transform then min-max normalize using log_min and log_max
            if feature == 'time':
                # we will log transform then min max it - note we need to add a window time feature when passing windows into the model.
                trajectories[:, j] = self.log_min_max(trajectories[:, j], self.obs_norm_dict[feature])  
            
            else:
                if feature in self.features_to_use['feature'].values:
                    feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
                    normalization_type = feature_row['normalization_type'].values[0]
                    
                else: 
                    if "mask" in feature:
                        normalization_type = 'none' # if it's a mask, it's binary, no need to normalize
                    elif "delta" in feature: 
                        normalization_type = 'min_max' # if it's delta, gemini said min-max it
                    else:
                        raise ValueError(f"Feature {feature} not found in features_to_use.csv, and it's not a mask or delta feature. Please check your features_to_use.csv file.")

                if normalization_type == 'none':
                    continue
                elif normalization_type == 'min_max':
                    trajectories[:, j] = self.min_max(trajectories[:, j], self.obs_norm_dict[feature])  
                elif normalization_type == 'log_min_max':
                    trajectories[:, j] = self.log_min_max(trajectories[:, j], self.obs_norm_dict[feature])
                elif normalization_type == 'z_score':
                    trajectories[:, j] = self.z_score(trajectories[:, j], self.obs_norm_dict[feature])
                elif normalization_type == 'robust_scaling':
                    trajectories[:, j] = self.robust_scale(trajectories[:, j], self.obs_norm_dict[feature])
                else:
                    raise ValueError(f"Invalid normalization type {normalization_type} for feature {feature}. Please check your features_to_use.csv file.")
               
            # for binary features, do nothing
            # throw error if any nans, infs, or valeus greater than 1e6
            if np.isnan(trajectories[:, j]).any():
                raise ValueError(f"NaNs found in normalized feature {feature} in shard {self.raw_shard_file}")
            if np.isinf(trajectories[:, j]).any():
                raise ValueError(f"Infs found in normalized feature {feature} in shard {self.raw_shard_file}")
            if np.abs(trajectories[:, j]).max() > 1e6:
                raise ValueError(f"Extreme values found in normalized feature {feature} in shard {self.raw_shard_file}")
           
        self.dynamic_deephit_dataset = {}
        self.dynamic_deephit_dataset['feature_names'] = feature_names
        self.dynamic_deephit_dataset['stay_ids'] = stay_ids
        self.dynamic_deephit_dataset['index_ptr'] = index_ptr
        self.dynamic_deephit_dataset['trajectories'] = trajectories.astype(np.float32)
        
        final_indices = index_ptr[1:] - 1    
        final_events = death_values[final_indices].astype(int)
                
        self.dynamic_deephit_dataset['event'] = final_events.astype(np.int32)

        logger.info(f"Finished normalizing shard file: {self.raw_shard_file}")

    
    def get_normalization_params(self):

        # check if self has the "obs_norm_dict" attribute
        if not hasattr(self, 'obs_norm_dict'):
            return None

        return {
            "obs_norm_dict": self.obs_norm_dict
        }   
    

    @ staticmethod
    def load_dynamic_deephit_dataset_from_h5(file_path, keys_to_load=None, keys_to_exclude=None):
        dynamic_deephit_dataset = {}

        if keys_to_load is None:
            keys_to_load = ['stay_ids', 'trajectories', 'index_ptr', 'feature_names', 'physiological_feature_ind', 
                            'physiological_feature_type', 'feature_dim', 'response_zero_ind', 'carry_forward_ind',
                            'event', 'time_to_event_raw', 'time_to_event']
        if keys_to_exclude is not None:
            keys_to_load = [key for key in keys_to_load if key not in keys_to_exclude]

        with h5py.File(file_path, 'r') as f:
            

            dynamic_deephit_dataset['stay_ids'] = f['stay_ids'][:]
            dynamic_deephit_dataset['trajectories'] = f['trajectories'][:].astype(np.float32)
            dynamic_deephit_dataset['index_ptr'] = f['index_ptr'][:]
                        
            dynamic_deephit_dataset['feature_names'] = [f['feature_names'][i].decode('utf-8') for i in range(len(f['feature_names']))]
            dynamic_deephit_dataset['physiological_feature_ind'] = f['physiological_feature_ind'][:].astype(np.int32).tolist()
            dynamic_deephit_dataset['physiological_feature_type'] = [f['physiological_feature_type'][i].decode('utf-8') for i in range(len(f['physiological_feature_type']))]
            dynamic_deephit_dataset['feature_dim'] = f['feature_dim'][()]
            dynamic_deephit_dataset['response_zero_ind'] = f['response_zero_ind'][:].astype(np.int32).tolist()
            dynamic_deephit_dataset['carry_forward_ind'] = f['carry_forward_ind'][:].astype(np.int32).tolist()
            
            dynamic_deephit_dataset['event'] = f['event'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event_raw'] = f['time_to_event_raw'][:].astype(np.int32)
            dynamic_deephit_dataset['time_to_event'] = f['time_to_event'][:].astype(np.int32)

        return dynamic_deephit_dataset
    
    @ staticmethod
    def load_dynamic_deephit_predictions_from_h5(file_path, keys_to_load=None, keys_to_exclude=None):
        dynamic_deephit_predictions = {}

        if keys_to_load is None:
            keys_to_load = ['predictions', 'stay_ids', 'index_ptr', 'model_outputs', 'health_score', 'entropy', 'carry_forward_ind', 'response_zero_ind']
        if keys_to_exclude is not None:
            keys_to_load = [key for key in keys_to_load if key not in keys_to_exclude]

        with h5py.File(file_path, 'r') as f:
            if 'predictions' in keys_to_load:
                dynamic_deephit_predictions['predictions'] = f['predictions'][:].astype(np.float32)
            if 'stay_ids' in keys_to_load:
                dynamic_deephit_predictions['stay_ids'] = f['stay_ids'][:]
            if 'index_ptr' in keys_to_load:
                dynamic_deephit_predictions['index_ptr'] = f['index_ptr'][:]
            if 'model_outputs' in keys_to_load:
                dynamic_deephit_predictions['model_outputs'] = f['model_outputs'][:].astype(np.float32)
            if 'health_score' in keys_to_load:    
                dynamic_deephit_predictions['health_score'] = f['health_score'][:].astype(np.float32)
            if 'entropy' in keys_to_load:  
                dynamic_deephit_predictions['entropy'] = f['entropy'][:].astype(np.float32)
            if 'carry_forward_ind' in keys_to_load:   
                dynamic_deephit_predictions['carry_forward_ind'] = f['carry_forward_ind'][:].astype(np.int32)
            if 'response_zero_ind' in keys_to_load:    
                dynamic_deephit_predictions['response_zero_ind'] = f['response_zero_ind'][:].astype(np.int32)


        
        return dynamic_deephit_predictions
    
   
   

        
def main():
    pass

if __name__ == "__main__":
    main()