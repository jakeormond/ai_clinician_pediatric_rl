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
                 config, 
                 test_size=0.2, 
                 val_size=0.2,
                 normalize_obs = True,
                 normalize_actions = True,
                 transform_targets = True):
        
        self.config = config
        self.reward = config["reward"]
        self.flip_reward = config["flip_reward"]

        self.actions = config["actions"]
        self.create_binary_actions = config["create_binary_actions"]
        self.transform_actions = config["transform_actions"]
        self.action_transforms = config["action_transforms"]

        self.proportion_w_data = config["proportions_to_use"]
        self.test_size = test_size
        self.val_size = val_size

        if self.val_size > 0.0:
            self.include_val = True
        else:
            self.include_val = False
        
        self.random_state = config["seed"]
        
        self, dataframe = self._load_dataset()
        self._split_and_process(dataframe)

        if self.create_binary_actions:
            self._create_binary_actions()

        if self.transform_actions: # this has to occur before normalization
            self._transform_actions()

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
                
            

    def _load_dataset(self):
        dataframe = pd.read_parquet(self.config["dataframe_path"])
        features_to_use_df = pd.read_csv(self.config["features_to_use_path"])

        dataframe = pd.read_parquet(self.config["dataframe_path"])
        
        features_to_use_df = pd.read_csv(self.config["features_to_use_path"])

        # drop terminal reward column if terminal_reward is False
        if not self.config["terminal_reward"]:
            if self.config["terminal_reward_col"] in features_to_use_df['feature'].values:
                features_to_use_df = features_to_use_df[features_to_use_df['feature'] != self.config["terminal_reward_col"]]
        
        # drop any rows where "do not use" is 1
        features_to_use_df = features_to_use_df[features_to_use_df['do not use'] != 1]

        # drop any rows where "proportion_sepsis" is less than proportion_w_data and
        # imputation_method is "interpolation"
        features_to_use_df = features_to_use_df[
            (features_to_use_df['proportion_sepsis'] >= self.proportion_w_data) |   
            (features_to_use_df['imputation_method'] != 'interpolation')
        ]

        # save lists of the observation features, actions, and rewards
        features = features_to_use_df['feature'].tolist()
        # drop self.action_features from obs_features
        self.obs_features = [f for f in features if f not in self.actions]
        # save the actions features

        # add to these the encounter_col, time_col, and the reward
        features_to_keep = [self.config["encounter_col"], self.config["time_col"], \
                                        self.config["reward"]] + features

        dataframe = dataframe[features_to_keep].copy()
        
        n_encounter_ids_before = dataframe[self.config["encounter_col"]].nunique()
        n_rows_before = dataframe.shape[0]  

        # remove rows with NaN values anywhere in the dataframe
        dataframe = dataframe.dropna()
        dataframe = dataframe.reset_index(drop=True)

        n_encounter_ids_after = dataframe[self.config["encounter_col"]].nunique()
        n_rows_after = dataframe.shape[0]

        print(f"Removed {n_encounter_ids_before - n_encounter_ids_after} "
                f"encounter IDs and {n_rows_before - n_rows_after} rows with NaN values.")
        print(f"Remaining encounter IDs: {n_encounter_ids_after}, Remaining rows: {n_rows_after}")       

        return self, dataframe

        
    def _split_and_process(self, dataframe):

        encounter_ids = dataframe[self.config["encounter_col"]].unique()

        # First split off the test set
        train_ids, test_ids = train_test_split(
            encounter_ids, test_size=self.test_size, 
            random_state=self.random_state
        )     

        if self.include_val:
            val_size_adjusted = self.val_size / (1 - self.test_size)

            train_ids, val_ids = train_test_split(
                train_ids, test_size=val_size_adjusted,
                random_state=self.random_state
            )

        else:
            val_ids = None

        train_data_dataframe = dataframe[dataframe[self.config["encounter_col"]].isin(train_ids)].copy()
        test_data_dataframe = dataframe[dataframe[self.config["encounter_col"]].isin(test_ids)].copy()
        if self.include_val:
            val_dataframe = dataframe[dataframe[self.config["encounter_col"]].isin(val_ids)].copy()
        else:
            val_dataframe = None

        # check for any all-zero columns in any of the dataframes
        zero_cols_train = [col for col in train_data_dataframe.columns if (train_data_dataframe[col] == 0).all()]
        zero_cols_test = [col for col in test_data_dataframe.columns if (test_data_dataframe[col] == 0).all()]
        if self.include_val:
            zero_cols_val = [col for col in val_dataframe.columns if (val_dataframe[col] == 0).all()]

        # if len(zero_cols_train) > 0 or len(zero_cols_test) > 0:
        #     raise ValueError(
        #         f"Found all-zero columns in train or test data: "
        #         f"train: {zero_cols_train}, test: {zero_cols_test}"
        #     )

        drop_cols = set(zero_cols_train + zero_cols_test)
        if self.include_val:
            drop_cols.update(zero_cols_val)

        if len(drop_cols) > 0:
            print(f"Dropping all-zero columns: {drop_cols}")
            train_data_dataframe = train_data_dataframe.drop(columns=drop_cols)
            test_data_dataframe = test_data_dataframe.drop(columns=drop_cols)
            if self.include_val:
                val_dataframe = val_dataframe.drop(columns=drop_cols)

        # split the dataframes into the arrays needed for RL
        self.train_data = self._split_into_rl_dataset(train_data_dataframe)
        self.test_data = self._split_into_rl_dataset(test_data_dataframe)
        if self.include_val:
            self.val_data = self._split_into_rl_dataset(val_dataframe)
        else:
            self.val_data = None

        
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
    

    def _create_binary_actions(self):
        
        self.train_data['binary_actions'] = self.train_data['actions'].copy()
        self.test_data['binary_actions'] = self.test_data['actions'].copy()
        if self.include_val:
            self.val_data['binary_actions'] = self.val_data['actions'].copy()        
        
        for i, action in enumerate(self.actions):
            # create a binary action for each action, where 1 means the action was given
            # and 0 means it was not given
            self.train_data['binary_actions'][:, i] = \
                (self.train_data['binary_actions'][:, i] != 0).astype(int)
            self.test_data['binary_actions'][:, i] = \
                (self.test_data['binary_actions'][:, i] != 0).astype(int)
            if self.include_val:
                self.val_data['binary_actions'][:, i] = \
                    (self.val_data['binary_actions'][:, i] != 0).astype(int)


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


    def _transform_actions(self):

        n_targets = len(self.actions)

        for i, target in enumerate(self.actions):
            target_transform = self.config["action_transforms"][i]

            if target_transform == "yeo-johnson": # mo other transforms yet implemented
                pt = PowerTransformer(method='yeo-johnson')

                train_target_data = self.train_data['actions'][:, i]
                train_target_no_zeros = train_target_data[train_target_data != 0]
                n_feature_rows = len(train_target_no_zeros)
                n_rows = len(train_target_data)
                print(f"Feature: {target}, Rows with data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

                # fit to the non-zero data in the training set
                pt.fit(train_target_no_zeros.reshape(-1, 1))

                # transform the training data
                self.train_data['actions'][:, i] = pt.transform(
                    self.train_data['actions'][:, i].reshape(-1, 1)).flatten()
                # transform the test data
                self.test_data['actions'][:, i] = pt.transform(
                    self.test_data['actions'][:, i].reshape(-1, 1)).flatten()
                if self.include_val:
                    self.val_data['actions'][:, i] = pt.transform(
                        self.val_data['actions'][:, i].reshape(-1, 1)).flatten()
                    
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