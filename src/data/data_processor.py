"""
Creates DataProcessor class to process data for RL training.

to do: check the vis scores, i think there are some outliers that are far too high. 

"""



import yaml
import pandas as pd
import numpy as np
import time
import pyarrow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

class DataProcessor:
    def __init__(self,
                 dataset, 
                 normalize_obs = True,
                 normalize_actions = True,
                 transform_targets = True):
        
        # self should inherit all the key value pairs from the dataset except the dataframe
        for key, value in vars(dataset).items():
            if key != "dataframe":
                setattr(self, key, value)
        
        if self.create_binary_actions:
            self._create_binary_actions(dataset)

        if self.transform_actions: # this has to occur before normalization
            self._transform_actions(dataset)

        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions

        if self.normalize_obs or self.normalize_actions:
            self._normalize_data()

        # combine actions and binary actions
        if self.create_binary_actions:
            self.train_data['actions'] = \
                np.concatenate((self.train_data['actions'], 
                           self.train_data['binary_actions']), axis=1)
            
            # remove the binary actions from train_data
            self.train_data.pop('binary_actions', None)


            self.test_data['actions'] = \
                np.concatenate((self.test_data['actions'], 
                           self.test_data['binary_actions']), axis=1)
            self.test_data.pop('binary_actions', None)

            if self.include_val:
                self.val_data['actions'] = \
                    np.concatenate((self.val_data['actions'], 
                               self.val_data['binary_actions']), axis=1) 
                self.val_data.pop('binary_actions', None)      
                
            
            
        
    def _split_into_rl_dataset(self, dataframe):
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

        rewards = dataframe[self.config["reward"]].copy()    
        # process the rewards
        if self.flip_reward:
            rewards = -rewards

        # now compute hourly change in reward
        rewards = rewards.diff().fillna(0)
        first_indices = dataframe.groupby(self.config["encounter_col"]).head(1).index
        rewards.loc[first_indices] = 0        
        
        actions = dataframe[self.config["actions"]].copy()
        observations = dataframe.drop(columns=[self.config["reward"]] + self.config["actions"]).copy()

        # compute terminated
        # terminate is 0 everywhere except the final timestep for each patient
        dataframe = dataframe.copy()
        dataframe['terminated'] = 0
        dataframe.loc[dataframe.groupby(self.config["encounter_col"]).tail(1).index, 'terminated'] = 1
        terminated = dataframe['terminated'].copy()

        # get the next observations
        next_observations = observations.copy()

        # get the indices of all the trajectory starts, i.e. the first index of each patient
        trajectory_starts = dataframe.groupby(self.config["encounter_col"]).head(1).index

        # get the trajectory ends, i.e. the last index of each patient
        trajectory_ends = dataframe.groupby(self.config["encounter_col"]).tail(1).index

        # for observations and actions, we remove the trajectory ends
        observations = observations.drop(index=trajectory_ends)
        actions = actions.drop(index=trajectory_ends)

        # for rewards, next_observations, and terminated, we remove the trajectory starts
        rewards = rewards.drop(index=trajectory_starts)
        next_observations = next_observations.drop(index=trajectory_starts)
        terminated = terminated.drop(index=trajectory_starts)

        # remove the encounter and time columns from observations and next_observations 
        observations = observations.drop(columns=[self.config["encounter_col"], 
                    self.config["time_col"]])
        next_observations = next_observations.drop(columns=[self.config["encounter_col"], 
                        self.config["time_col"]])

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
            dataset.dataframe[binary_col_name] = (dataset.dataframe[action] != 0).astype(int)



    def _normalize_data(self):
        # FIX THIS CODE!!!!!!!!!!!!!!!!!!!!!!!
        # We should apply min max normalization to all drugs!!!!!!!!

        # This function will normalize the observations using the mean and std of the training set
        # It will return the normalized observations (centred at 0)
        if self.normalize_obs:
            self.mean_obs = self.train_data['observations'].mean(axis=0)
            self.std_obs = self.train_data['observations'].std(axis=0)

            self.train_data['observations'] = (self.train_data['observations'] 
                                               - self.mean_obs) / self.std_obs
            self.test_data['observations'] = (self.test_data['observations'] 
                                              - self.mean_obs) / self.std_obs
            if self.include_val:
                self.val_data['observations'] = (self.val_data['observations'] 
                                                 - self.mean_obs) / self.std_obs
                
            print("Observation normalization complete.")
        
        if self.normalize_actions: # these should be 0 to 1, since we
            # have many zeros. Note min can be negative because of transforms
            self.min_actions = self.train_data['actions'].min(axis=0)            
            self.max_actions = self.train_data['actions'].max(axis=0)

            self.train_data['actions'] = (self.train_data['actions'] - self.min_actions) / \
                                          (self.max_actions - self.min_actions)

            self.test_data['actions'] = (self.test_data['actions'] - self.min_actions) / \
                                          (self.max_actions - self.min_actions)
            if self.include_val:
                self.val_data['actions'] = (self.val_data['actions'] - self.min_actions) / \
                                          (self.max_actions - self.min_actions)
                  
            print("Action normalization complete.")


    def _transform_actions(self, dataset):

        n_targets = len(self.actions)

        # ✅ Store the fitted transformers for later use
        self.action_transformers = {}

        for i, target in enumerate(self.actions):
            target_transform = self.action_transforms[i]

            if target_transform == "yeo-johnson": # no other transforms yet implemented
                pt = PowerTransformer(method='yeo-johnson')                 

                # fit to the non-zero data in the training set
                pt.fit(dataset.dataframe[target].values.reshape(-1, 1))  # Reshape to 2D array
                self.action_transformers[target] = pt

                # transform the training data
                dataset.dataframe[target] = pt.transform(
                    dataset.dataframe[target].values.reshape(-1, 1)).flatten()
                                   
                print(f"Transformed target {target} with Yeo-Johnson transform.")
                

def main():
    hospital = 'choc'  # or 'mimic'
    config_path = f"config/{hospital}.yaml"
    config = load_config(config_path)

    tick = time.time()
    processed_data = DataProcessor(config, 
                    test_size=0.2, val_size=0.0,)
    tock = time.time()
    print(f"Data processing took {tock - tick:.2f} seconds.")
    
    pass

if __name__ == "__main__":
    main()