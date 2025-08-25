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

class DataProcessor:
    def __init__(self, dataset, training_config, norm_params=None, 
                 normalize_obs = True, normalize_actions = True,
                 transform_actions = True, create_binary_actions = True):
                
        # self should inherit all the key value pairs from the dataset except the dataframe
        for key, value in vars(dataset).items():
            if key != "dataframe":
                if isinstance(value, (list, dict, set)):
                    # Deep copy mutable objects
                    setattr(self, key, copy.deepcopy(value))
                else:
                    # Shallow copy is fine for immutable objects (str, int, float, etc.)
                    setattr(self, key, value)

        # self should also inherit all the key value pairs from the training_config
        for key, value in training_config.items():
            setattr(self, key, value)
              
        # drop any features in self.features_to_drop
        for feature in self.features_to_drop:
            if feature in dataset.dataframe.columns:
                dataset.dataframe = dataset.dataframe.drop(columns=[feature])
        
        # before creating the binary actions, we need to make a list of all the input features;
        # these are the columns minus the reward, the actions and the encounter and time columns
        self.input_features = [col for col in dataset.dataframe.columns
                               if col not in dataset.actions and 
                               col != self.reward and 
                               col != self.encounter_col and 
                               col != self.time_col]
        
        for f in self.input_features:
            print(f"Input feature: {f}")

        # split into the rl_dataset
        self.rl_dataset = self._split_into_rl_dataset(dataset)           
            
        self.norm_params = norm_params
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.transform_actions = transform_actions

        if self.normalize_obs or self.normalize_actions or self.transform_actions:
            self.normalize()

        if create_binary_actions: # note that this requires zeros to have remained zeros after transformation and normalization
            self._create_binary_actions()


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
            
        
    def _split_into_rl_dataset(self, dataset):
        # This function will split the dataframe into the format needed for RL
        # It will return a dictionary with the following keys:
        # - observations
        # - actions
        # - rewards
        # - terminations
        # - next_observations
        # 
        # state and action come from a given timestep, reward, next_state and terminated 
        # come from the next timestep.
        # Therefore, each trajectory will actually be length n_timesteps - 1. So observations
        # and actions will have indices 0 to n_timesteps - 2, next_observations and rewards 
        # and terminated will have indices 1 to n_timesteps - 1.

        # we piece off the rewards and actions and then will have the observations left

        ##### SHOULD ADD SOME KIND OF UNIT TEST HERE TO CONFIRM EVERYTHING IS WORKING AS EXPECTED #####

        rewards = dataset.dataframe[self.reward].copy()    
        # process the rewards
        if self.flip_reward:
            rewards = -rewards

        # now compute hourly change in reward
        rewards = rewards.diff().fillna(0)
        first_indices = dataset.dataframe.groupby(self.encounter_col).head(1).index
        rewards.loc[first_indices] = 0        
        
        actions = dataset.dataframe[self.actions].copy()
        observations = dataset.dataframe[self.input_features].copy()

        # compute terminated
        # terminate is 0 everywhere except the final timestep for each patient
        dataset.dataframe['terminated'] = 0
        dataset.dataframe.loc[dataset.dataframe.groupby(self.encounter_col).tail(1).index, 'terminated'] = 1
        terminated = dataset.dataframe['terminated'].copy()
        # drop the terminated column from the dataframe
        dataset.dataframe = dataset.dataframe.drop(columns=['terminated'])

        # get the next observations
        next_observations = observations.copy()

        # get the indices of all the trajectory starts, i.e. the first index of each patient
        trajectory_starts = dataset.dataframe.groupby(self.encounter_col).head(1).index

        # get the trajectory ends, i.e. the last index of each patient
        trajectory_ends = dataset.dataframe.groupby(self.encounter_col).tail(1).index

        # for observations and actions, we remove the trajectory ends
        observations = observations.drop(index=trajectory_ends)
        actions = actions.drop(index=trajectory_ends)

        # for rewards, next_observations, and terminated, we remove the trajectory starts
        rewards = rewards.drop(index=trajectory_starts)
        next_observations = next_observations.drop(index=trajectory_starts)
        terminated = terminated.drop(index=trajectory_starts)

        # convert to numpy arrays
        observations = observations.to_numpy()  
        actions = actions.to_numpy()
        rewards = rewards.to_numpy()
        next_observations = next_observations.to_numpy()
        terminated = terminated.to_numpy()

        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "next_observations": next_observations,
            "terminated": terminated
        }
    

    def _create_binary_actions(self):
                
        # loop through the columns in self.rl_dataset['actions'] and create a
        # binary action for each action, where 1 means the action was given
        # and 0 means it was not given, adding the column onto the array       
    
        actions_array = self.rl_dataset['actions']
        
        # Vectorized binary creation for all columns at once
        binary_actions = (actions_array > 0).astype(int)
        
        # Concatenate original and binary actions
        self.rl_dataset['actions'] = np.concatenate([actions_array, binary_actions], axis=1)
        
        # Update action names
        original_names = self.actions.copy()  # Preserve original names
        binary_names = [f"{name}_binary" for name in original_names]
        self.actions.extend(binary_names)
        
        print(f"Created {binary_actions.shape[1]} binary actions")
        print(f"Original actions: {original_names}")
        print(f"Binary actions: {binary_names}")
        print(f"New action array shape: {self.rl_dataset['actions'].shape}")


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
                self.rl_dataset['observations'][:, i] = (self.rl_dataset['observations'][:, i] - 
                    self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                                                           - self.obs_norm_dict[feature]['min'])
                
                # it is possible here that if both min and max are 0, we will get NaNs;
                # in this case, we set the normalized value to 0
                self.rl_dataset['observations'][:, i] = \
                    np.nan_to_num(self.rl_dataset['observations'][:, i], nan=0.0)
                
                # also normalize the next_observations
                self.rl_dataset['next_observations'][:, i] = (self.rl_dataset['next_observations'][:, i] - 
                    self.obs_norm_dict[feature]['min']) / (self.obs_norm_dict[feature]['max'] 
                                                           - self.obs_norm_dict[feature]['min'])     
                self.rl_dataset['next_observations'][:, i] = \
                    np.nan_to_num(self.rl_dataset['next_observations'][:, i], nan=0.0)     

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

            print(f"Normalized feature {feature} with {normalization_type} normalization. "
                f"Min: {self.obs_norm_dict[feature]['min']}, \
                    Max: {self.obs_norm_dict[feature]['max']}, \
                        Mean: {self.obs_norm_dict[feature]['mean']}, \
                            Std: {self.obs_norm_dict[feature]['std']}")

        print("Observation normalization complete.")                        

    
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

            self.rl_dataset['actions'][:, i] = (self.rl_dataset['actions'][:, i] - self.action_minimum[action]) / \
                                              (self.action_maximum[action] - self.action_minimum[action])
            

            print(f"Normalized action {action} between 0 and 1.")        


    def _transform_actions(self):

        # Store the fitted transformers for later use
        if self.action_transformers is None:
            self.action_transformers = {}

        for i, target in enumerate(self.actions):
            target_transform = self.action_transforms[i]

            if target_transform == "yeo-johnson": # no other transforms yet implemented
                
                if target not in self.action_transformers:
                    # fit the transformer on the training data
                    pt = PowerTransformer(method='yeo-johnson')    
                    pt.fit(self.rl_dataset['actions'][:, i].reshape(-1, 1))  # Reshape to 2D array
                    self.action_transformers[target] = pt  

                else:
                    pt = self.action_transformers[target]            

                # transform the training data
                self.rl_dataset['actions'][:, i] = pt.transform(
                    self.rl_dataset['actions'][:, i].reshape(-1, 1)).flatten()
                                   
                print(f"Transformed target {target} with Yeo-Johnson transform.")

    
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