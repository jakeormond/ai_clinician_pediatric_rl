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
    def __init__(self, dataset, training_config, norm_params=None):
                
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

        if norm_params is not None:
            # if normalization parameters are provided, use them to normalize the data
            self.obs_norm_dict = norm_params['obs_norm_dict']
            self.action_minimum = norm_params['action_minimum']
            self.action_maximum = norm_params['action_maximum']
            self.action_transformers = norm_params['action_transformers']
        else:
            self.obs_norm_dict = None
            self.action_minimum = None
            self.action_maximum = None
            self.action_transformers = None
        
        if self.normalize_obs:
            self._normalize_obs(dataset)               

        if self.transform_actions: # this has to occur before normalization
            self._transform_actions(dataset)

        if self.normalize_actions:
            self._normalize_actions(dataset)
        
        if self.create_binary_actions: # note that this requires zeros to have remained zeros after transformation and normalization
            self._create_binary_actions(dataset)

        # split into the rl_dataset
        self.rl_dataset = self._split_into_rl_dataset(dataset)               
            
        
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
    

    def _create_binary_actions(self, dataset):
                
        for i, action in enumerate(dataset.actions):
            # create a binary action for each action, where 1 means the action was given
            # and 0 means it was not given
            binary_col_name = f"{action}_binary"
            self.actions.append(binary_col_name)
            dataset.dataframe[binary_col_name] = (dataset.dataframe[action] != 0).astype(int)



    def _normalize_obs(self, dataset):
        # apply min-max normalization to drugs and ordinal features, 
        # leave one hot encoded features as they are, and z-score normalize 
        # the continuous features

        norm_dict = {}

        for feature in self.input_features:
            
            norm_dict[feature] = {}
            
            # get the features row in self.features_to_use
            feature_row = self.features_to_use[self.features_to_use['feature'] == feature]
            data_type = feature_row['data_type'].values[0]
            feature_set = feature_row['feature_set'].values[0]

            min_val = dataset.dataframe[feature].min()
            max_val = dataset.dataframe[feature].max()
            mean_val = dataset.dataframe[feature].mean()
            std_val = dataset.dataframe[feature].std()
            
            if data_type == 'ordinal' or feature_set == 'infusion' or \
                feature_set == 'other_meds' or feature_set == 'sedation':
                
                # min-max normalize the feature
                norm_dict[feature]['normalization_type'] = 'min-max'
                dataset.dataframe.loc[:, feature] = (dataset.dataframe[feature] - min_val) / (max_val - min_val)                

            elif data_type == 'binary':
                # binary features are already normalized, so we just store the normalization type
                norm_dict[feature]['normalization_type'] = 'none'

            elif data_type == 'continuous':
                # z-score normalize the feature
                norm_dict[feature]['normalization_type'] = 'z-score'
                dataset.dataframe.loc[:, feature] = (dataset.dataframe[feature] - mean_val) / std_val

            norm_dict[feature]['min'] = min_val
            norm_dict[feature]['max'] = max_val
            norm_dict[feature]['mean'] = mean_val
            norm_dict[feature]['std'] = std_val

            print(f"Normalized feature {feature} with {norm_dict[feature]['normalization_type']} normalization. "
                  f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")


        # store the normalization dict for later use
        self.obs_norm_dict = norm_dict

        print("Observation normalization complete.")                        

    
    def _normalize_actions(self, dataset):
        # normalize actions between 0 and 1
        # will need to store the min and max for later use
        self.action_minimum = {}
        self.action_maximum = {}

        for i, action in enumerate(dataset.actions):
            # normalize each action between 0 and 1
            # we will use the min and max of the training set to normalize the test set
            min_action = dataset.dataframe[action].min()
            max_action = dataset.dataframe[action].max()

            dataset.dataframe.loc[:, action] = (dataset.dataframe[action] - min_action) / \
                                                  (max_action - min_action) 

            self.action_minimum[action] = min_action
            self.action_maximum[action] = max_action     

            print(f"Normalized action {action} between 0 and 1.")        


    def _transform_actions(self, dataset, transform_included=False):

        # Store the fitted transformers for later use
        self.action_transformers = {}

        for i, target in enumerate(self.actions):
            target_transform = self.action_transforms[i]

            if target_transform == "yeo-johnson": # no other transforms yet implemented
                
                if not transform_included:
                    pt = PowerTransformer(method='yeo-johnson')    
                    pt.fit(dataset.dataframe[target].values.reshape(-1, 1))  # Reshape to 2D array
                    self.action_transformers[target] = pt  

                else:
                    pt = self.action_transformers[target]            

                # transform the training data
                dataset.dataframe[target] = pt.transform(
                    dataset.dataframe[target].values.reshape(-1, 1)).flatten()
                                   
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