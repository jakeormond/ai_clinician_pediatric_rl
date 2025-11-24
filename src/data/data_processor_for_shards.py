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
            verbose=False
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

        if config_manager.action_mode == 'binary_only' or config_manager.action_mode == 'mixed':
            create_binary_actions = True
        else:
            create_binary_actions = False            

        if config_manager.action_mode == 'binary_only':
            self.normalize_actions = False
            self.transform_actions = False           


        # find the shards for the given split
        self.shard_dir = config_manager.data_config['shard_dir']
        all_files = os.listdir(self.shard_dir)
        raw_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                f.startswith(self.split) and f.endswith("_shard.h5")]
        # sort the files to ensure consistent order
        self.raw_shard_files = sorted(raw_shard_files)

        if normalize:
            normalized_shard_files = [os.path.join(self.shard_dir, f) for f in all_files if 
                                    f.startswith(self.split) and f.endswith("_shard_normalized.h5")]
            # sort the files to ensure consistent order
            self.normalized_shard_files = sorted(normalized_shard_files)

            if len(self.normalized_shard_files) == 0: # i.e. the files have not been normalized yet
                self.normalize()


    def normalize(self):

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

        for shard_file in self.raw_shard_files:
            logger.info(f"Normalizing shard file: {shard_file}")
            # load the shard file
            with h5py.File(shard_file, 'r') as f:
                # load the dataframe
                df = pd.read_hdf(shard_file, key='dataframe')
        
        
                # List all groups/datasets
                print(list(f.keys()))
                
                # Access a dataset
                # dataset = f['dataset_name'][:]  # Use [:] to read the data into a NumPy array
                # print(dataset.shape, dataset.dtype)
        
        
        
        
        
        
        
        if dataset.normalize_obs:
            self._normalize_obs(dataset)               

        if dataset.transform_actions: # this has to occur before normalization
            self._transform_actions()

        if dataset.normalize_actions:
            self._normalize_actions() 
            
        
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

            # # additonally, we need to train on partial windows from the start of the trajectory
            # if traj_length > self.window_length: # then we need to sample additional partial windows from trajectory start
            #     early_window_end_indices = np.random.choice(
            #         np.arange(0, self.window_length), 
            #         size=min(n_early_windows, self.window_length), 
            #         replace=False).tolist()
            #     window_ind += early_window_end_indices

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
                    # window_features = np.pad(window_features, 
                    #                         ((pad_width, 0), (0, 0)), 
                    #                         mode='constant', 
                    #                         constant_values=0)
                    # mask_array[array_ind, pad_width:] = 1
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