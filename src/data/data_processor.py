"""
Creates DataProcessor class to process data for RL training.
Note all normalization and transformation is done here, not in the 
preceding dataset.py file (that's because Dataset is only used
for supervised learning with XGBoost, which does not require
normalization).

to do: check the vis scores, i think there are some outliers that are far too high. 

"""
import yaml
import copy
import pandas as pd
import numpy as np
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

import torch
from torch.utils.data import Dataset, DataLoader

import logging
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, dataset, config_manager, norm_params=None, verbose=False):
                
        # self should inherit all the key value pairs from the dataset except the dataframe
        for key, value in vars(dataset).items():
            if key != "dataframe":
                if isinstance(value, (list, dict, set)):
                    # Deep copy mutable objects
                    setattr(self, key, copy.deepcopy(value))
                else:
                    # Shallow copy is fine for immutable objects (str, int, float, etc.)
                    setattr(self, key, value)
             
        
        if config_manager.action_mode == 'binary_only' or config_manager.action_mode == 'mixed':
            create_binary_actions = True
        else:
            create_binary_actions = False            

        if config_manager.action_mode == 'binary_only':
            self.normalize_actions = False
            self.transform_actions = False           

        ########### I THINK THERE IS NO REASON TO COPY THE TRAINING CONFIG INTO SELF ####################
        ############# ONCE THIS IS CONFIRMED, DELETE THE COMMENTED OUT CODE ##############################
        # # self should also inherit all the key value pairs from the training_config
        # for key, value in training_config.items():
        #     setattr(self, key, value)
              
        # drop any features in self.features_to_drop
        for feature in self.features_to_drop:
            if feature in dataset.dataframe.columns:
                dataset.dataframe = dataset.dataframe.drop(columns=[feature])
        
        # before creating the binary actions, we need to make a list of all the input features;
        # these are the columns minus the reward, the actions and the encounter and time columns
        if self.model_type == "supervised_learning":
            self.input_features = [col for col in dataset.dataframe.columns
                                if col != self.encounter_col and 
                                col != self.time_col and 
                                col != 'mortality' and 
                                col != 'discharge']
        else:
            self.input_features = [col for col in dataset.dataframe.columns
                                if col not in dataset.actions and 
                                col != self.reward_col and 
                                col != self.encounter_col and 
                                col != self.time_col and 
                                col != self.terminal_reward_col]
        
        for f in self.input_features:
            logger.debug(f"Input feature: {f}")

        if self.model_type != "dynamic_deephit":
            # split into dymamic deephit dataset
            self.dynamic_deephit_dataset = self._split_into_dynamic_deephit_dataset(dataset, config_manager)
        else:
            # split into the rl_dataset
            self.rl_dataset = self._split_into_rl_dataset(dataset, config_manager)           
                
        self.norm_params = norm_params
        # self.normalize_obs = normalize_obs
        # self.normalize_actions = normalize_actions
        # self.transform_actions = transform_actions

        if self.normalize_obs or self.normalize_actions or self.transform_actions:
            self.normalize()

        if create_binary_actions: # note that this requires zeros to have remained zeros after transformation and normalization
            self._create_binary_actions(config_manager)

    
    def get_start_states(self):
        """Get starting states for V0 estimation."""
        start_ind = np.where(self.rl_dataset["terminated"] == 1)[0] + 1
        start_ind = start_ind[start_ind < len(self.rl_dataset["terminated"])]
        start_ind = np.insert(start_ind, 0, 0)  # add the very first index

        starting_states = self.rl_dataset["observations"][start_ind]
        # make starting states a tensor so it can be passed to the actor and critic networks
        starting_states = torch.tensor(starting_states, dtype=torch.float32)
        # move to CUDA device
        starting_states = starting_states.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return starting_states


    def normalize(self, norm_params=None, normalize_obs=True,
                  normalize_actions=True, transform_actions=True):

        if self.norm_params is None and norm_params is not None:
            self.norm_params = norm_params
        
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
        
        if normalize_obs:
            self._normalize_obs()               

        if transform_actions: # this has to occur before normalization
            self._transform_actions()

        if normalize_actions:
            self._normalize_actions() 
            
        
    def _split_into_dynamic_deephit_dataset(self, dataset, config_manager):
        """
        This function will split the dataframe into the format needed for dynamic deephit
        It will return a dictionary with the following keys:
        - 
        """
        self.horizon = config_manager.training_config['horizon']

        # get all trajectory lengths
        traj_lengths = dataset.dataframe.groupby(self.encounter_col).size().to_dict()
        
        # calculate n_windows as the traj_length divided by horizon, rounded up
        n_windows = {k: int(np.ceil(v / self.horizon)) for k, v in traj_lengths.items()}
        
        # for each encounter_id, pick n_windows start indices randomly as integers between
        # 0 and traj_length
        start_indices = {}
        for enc_id, n_win in n_windows.items():
            traj_length = traj_lengths[enc_id]
            # pick n_win random integers between 0 and traj_length - 1
            starts = np.random.choice(range(traj_length), size=n_win, replace=False)
            start_indices[enc_id] = starts

        # 





        
        # truncate each patient trajectory (by encounter_id) to the horizon
        truncated_df = dataset.dataframe.groupby(self.encounter_col).tail(self.horizon).reset_index(drop=True)

        # create new column 'mask' and populate with 1s
        truncated_df['mask'] = 1

        # create a time_to_event column that counts from 1 to trajectory length for each patient
        truncated_df['time_to_event'] = truncated_df.groupby(self.encounter_col).cumcount() + 1

        # get the lengths of each trajectory
        traj_lengths = truncated_df.groupby(self.encounter_col).size().to_dict()

        # restrict traj_lengths dictionary to only those with length < horizon
        traj_lengths = {k: v for k, v in traj_lengths.items() if v < self.horizon}

        for enc_id, length in traj_lengths.items():
            # insert rows of zeros before the first rows to path the trajectory to the horizon
            rows_to_add = self.horizon - length
            first_index = truncated_df[truncated_df[self.encounter_col] == enc_id].index[0]
            zero_rows = pd.DataFrame(0, index=np.arange(rows_to_add), columns=truncated_df.columns)
            # make the encounter_col = enc_id for these rows
            zero_rows[self.encounter_col] = enc_id
            # insert the zero rows into the dataframe
            truncated_df = pd.concat([truncated_df.iloc[:first_index], zero_rows, truncated_df.iloc[first_index:]]).reset_index(drop=True)
           
        pass



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


    def _normalize_obs(self):
        # apply min-max normalization to drugs and ordinal features, 
        # leave one hot encoded features as they are, and z-score normalize 
        # the continuous features

        if self.obs_norm_dict is None:
            self.obs_norm_dict = {}

        for i, feature in enumerate(self.input_features):
                    
            # get the features row in self.features_to_use
            feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
            data_type = feature_row['data_type'].values[0]
            feature_set = feature_row['feature_set'].values[0]

            if feature not in self.obs_norm_dict:
                self.obs_norm_dict[feature] = {}
                self.obs_norm_dict[feature]['min'] = self.rl_dataset['observations'][:, i].min()
                self.obs_norm_dict[feature]['max'] = self.rl_dataset['observations'][:, i].max()
                self.obs_norm_dict[feature]['mean'] = self.rl_dataset['observations'][:, i].mean()
                self.obs_norm_dict[feature]['std'] = self.rl_dataset['observations'][:, i].std()
            
            if data_type == 'ordinal' or feature_set == 'infusion' or \
                feature_set == 'other_meds' or feature_set == 'sedation':
                
                # min-max normalize the feature
                normalization_type = 'min-max'
                
                # normalize the ith col in self.rl_dataset['observations']
                # using the min and max from the training set
                max_min_diff = self.obs_norm_dict[feature]['max'] - self.obs_norm_dict[feature]['min']
                if max_min_diff == 0:
                    self.rl_dataset['observations'][:, i] = 0.0
                    self.rl_dataset['next_observations'][:, i] = 0.0                
                else:
                    self.rl_dataset['observations'][:, i] = (self.rl_dataset['observations'][:, i] - 
                        self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                                                            - self.obs_norm_dict[feature]['min'])
                
                    # it is possible here that if both min and max are 0, we will get NaNs;
                    # in this case, we set the normalized value to 0
                    self.rl_dataset['observations'][:, i] = \
                        np.nan_to_num(self.rl_dataset['observations'][:, i], nan=0.0)
                
                    # it is possible to get very large values that are less than inf, set them to zero
                    self.rl_dataset['observations'][:, i] = \
                        np.where(np.abs(self.rl_dataset['observations'][:, i]) > 1e6, 0.0, self.rl_dataset['observations'][:, i])                
                    
                    # also possible to get Infs if max == min but numerator is non-zero, so we set those to 0 as well
                    self.rl_dataset['observations'][:, i] = \
                        np.where(np.isinf(self.rl_dataset['observations'][:, i]), 0.0, self.rl_dataset['observations'][:, i])
                    
                    # also normalize the next_observations
                    self.rl_dataset['next_observations'][:, i] = (self.rl_dataset['next_observations'][:, i] - 
                        self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                                                            - self.obs_norm_dict[feature]['min'])    

                    self.rl_dataset['next_observations'][:, i] = \
                        np.where(np.abs(self.rl_dataset['next_observations'][:, i]) > 1e6, 0.0, self.rl_dataset['next_observations'][:, i])

                    self.rl_dataset['next_observations'][:, i] = \
                        np.nan_to_num(self.rl_dataset['next_observations'][:, i], nan=0.0)   

                    self.rl_dataset['next_observations'][:, i] = \
                        np.where(np.isinf(self.rl_dataset['next_observations'][:, i]), 0.0, self.rl_dataset['next_observations'][:, i])  

            elif data_type == 'binary':
                # binary features are already normalized, so we just store the normalization type
                normalization_type = 'none'

            elif data_type == 'continuous':
                # z-score normalize the feature
                normalization_type = 'z-score'

                # normalize the ith col in self.rl_dataset['observations']
                # using the mean and std from the training set
                self.rl_dataset['observations'][:, i] = (self.rl_dataset['observations'][:, i] - 
                    self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']
                
                # also normalize the next_observations
                self.rl_dataset['next_observations'][:, i] = (self.rl_dataset['next_observations'][:, i] - 
                    self.obs_norm_dict[feature]['mean']) / self.obs_norm_dict[feature]['std']

            # check for NaNs in the normalized feature
            if np.isnan(self.rl_dataset['observations'][:, i]).any() or \
               np.isnan(self.rl_dataset['next_observations'][:, i]).any():
                raise ValueError(f"NaNs found in normalized feature {feature}")

            # check for Infs in the normalized feature
            if np.isinf(self.rl_dataset['observations'][:, i]).any() or \
               np.isinf(self.rl_dataset['next_observations'][:, i]).any():
                raise ValueError(f"Infs found in normalized feature {feature}") 

            logger.debug(f"Normalized feature {feature} with {normalization_type} normalization. "
                f"Min: {self.obs_norm_dict[feature]['min']}, \
                    Max: {self.obs_norm_dict[feature]['max']}, \
                        Mean: {self.obs_norm_dict[feature]['mean']}, \
                            Std: {self.obs_norm_dict[feature]['std']}")

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