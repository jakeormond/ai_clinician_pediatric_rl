import os
import pandas as pd
import copy
import numpy as np

# this class holds the complete dataset, the folds can 
# access the data they need using the encounter IDs
# and the methods in this class
class Dataset:
    def __init__(self, config, drop_actions=True, transform_target=True): # note that actions get dropped only for supervised learning, not RL
        
        # Dataset should store all the key value pairs from the config
        for key, value in config.items():
            setattr(self, key, value)
        
        self._load_dataset(drop_actions)
        
        if "target_col" in config and transform_target: # this is generally used for supervised learning, so gets
            self._transform_target() # accept dataframe as input


    def save_dataframe(self, config):

        loaded_df_path = config.data_config['dataframe_path']
        load_df_filename = os.path.basename(loaded_df_path)
        # strip .parquet from filename
        load_df_filename = load_df_filename.replace(".parquet", "")

        proportion_w_data = config.data_config['proportion_w_data']

        filename = f"{load_df_filename}_prop_{proportion_w_data}.parquet"
        path = config.get_experiments_dir + "/" + filename
        self.dataframe.to_parquet(path, index=False)
        print(f"DataFrame saved to {path}")


    def _load_dataset(self, drop_actions):
        self.dataframe = pd.read_parquet(self.dataframe_path) # here I will have to add code to load the df with the reward
        self.features_to_use = pd.read_csv(self.features_to_use_path)

         # drop any rows where "do not use" is 1
        self.features_to_use = self.features_to_use[self.features_to_use['do not use'] != 1]

        # if this is supervised learning, we need to drop the action features,
        # but if this is reinforcement learning, we need to keep them
        if self.model_type == "supervised_learning" and drop_actions:
            # get rid of actions
            self.features_to_use = self.features_to_use[self.features_to_use['state_or_action'] != 'action']
        
        # drop any rows where "proportion_sepsis" is less than proportion_w_data and
        # imputation_method is "interpolation"
        self.features_to_use = self.features_to_use[
            (self.features_to_use['proportion_sepsis'] >= self.proportion_w_data) |   
            (self.features_to_use['imputation_method'] != 'interpolation')
        ]

        # make list of features
        features = self.features_to_use['feature'].tolist()

        # append reward col, self.reward, to features if reinforcement learning
        if self.model_type == "reinforcement_learning":
            if self.reward:
                features.append(self.reward)

            if self.terminal_reward:
                if self.terminal_reward_col not in features:
                    features.append(self.terminal_reward_col)

        # add to these the encounter_col, time_col,
        features_to_keep = [self.encounter_col, self.time_col] + features

        self.dataframe = self.dataframe[features_to_keep].copy()
        
        n_encounter_ids_before = self.dataframe[self.encounter_col].nunique()
        n_rows_before = self.dataframe.shape[0]  

        # remove rows with NaN values anywhere in the dataframe
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        n_encounter_ids_after = self.dataframe[self.encounter_col].nunique()
        n_rows_after = self.dataframe.shape[0]

        print(f"Removed {n_encounter_ids_before - n_encounter_ids_after} "
                f"encounter IDs and {n_rows_before - n_rows_after} rows with NaN values.")
        print(f"Remaining encounter IDs: {n_encounter_ids_after}, Remaining rows: {n_rows_after}")       

        return self
    

    def _transform_target(self):
        """ depending on the type of prediction, we may need to alter
        the target data. For example, if we are predicting mortality,
        we may need to transform all the timepoints to a binary outcome
        (1 if the patient died, 0 if they survived). If we are predicting 
        mortality at a specific timepoint, we may need to transform the target
        to a binary outcome (1 if the patient died at that timepoint, 0 if 
        they survive) only for those specific timepoints.
        """

        encounter_ids = self.dataframe[self.encounter_col].unique()

        # find the encounter_ids that have target_data equal to 1
        target_encounter_ids = self.dataframe[self.dataframe[self.target_col] == 1][self.encounter_col].unique()
        print(f"Found {len(target_encounter_ids)} encounter IDs with target_data = 1")
        
        # get the surviving encounter_ids
        surviving_encounter_ids = set(encounter_ids) - set(target_encounter_ids)
        
        if self.prediction_type == "mortality_all":
            # all values for the target_data are 1 for the encounter_ids that have target_data equal to 1
            self.dataframe[self.target_col] = self.dataframe[self.encounter_col].apply(
                lambda x: 1 if x in target_encounter_ids else 0
            )       

        else:
            non_sur_data = self.dataframe[
                    self.dataframe[self.encounter_col].isin(target_encounter_ids)
                ].copy()
            
            if self.prediction_type == "mortality_24":
                # just the last 24 rows are labelled as 1 for the encounter_ids that have target_data equal to 1
                # Get data for encounters that should have mortality=1
                

                last_24_rows = non_sur_data.groupby(self.encounter_col).tail(24)

                # Set target to 1 for these rows
                self.dataframe.loc[last_24_rows.index, self.target_col] = 1  

            elif self.prediction_type == "discharge_mortality":                    
                
                # for each encounter in survivor_data, set the first half of rows to 1
                first_half_indices = (
                    self.dataframe.groupby(self.encounter_col)
                    .apply(lambda group: group.index[:len(group)//2])
                    .explode()
                    .astype(int)
                )

                # Set target to 1 for first half rows
                self.dataframe.loc[first_half_indices, self.target_col] = 1

                # set second half indices to 0 for non-survivors
                second_half_non_indices = (
                    non_sur_data.groupby(self.encounter_col)
                    .apply(lambda group: group.index[len(group)//2:])
                    .explode()
                    .astype(int)
                )
                self.dataframe.loc[second_half_non_indices, self.target_col] = 0

                # set second half indices to 2 for survivors
                survivor_data = self.dataframe[
                    self.dataframe[self.encounter_col].isin(surviving_encounter_ids)
                ].copy()

                second_half_sur_indices = ( 
                    survivor_data.groupby(self.encounter_col)
                    .apply(lambda group: group.index[len(group)//2:])
                    .explode()
                    .astype(int)
                )
                self.dataframe.loc[second_half_sur_indices, self.target_col] = 2

            else:
                # not implemented yet, so raise an error
                raise NotImplementedError("Prediction time 'specific' is not implemented yet.")
          

    def modeling_data(self, encounter_ids, split_x_y=False):
        """
        Returns the modeling data as a DataFrame.
        This is the data that will be used for training and testing.
        """
        
        modeling_data = self.dataframe[self.dataframe[self.encounter_col].isin(encounter_ids)].copy()
        
        encounters_and_times = modeling_data[[self.encounter_col, self.time_col]]

        modeling_data = modeling_data.drop(columns=[self.encounter_col, self.time_col]).copy()

        if split_x_y:
        
            X = modeling_data.drop(columns=[self.target_col]).copy()
            y = modeling_data[self.target_col].copy()
                
            return X, y, encounters_and_times   
        
        else:
            return modeling_data, encounters_and_times
        

    
    def encounter_ids(self):
        """
        Returns the unique encounter IDs in the dataset.
        """
        return self.dataframe[self.encounter_col].unique()
    

    def create_subset(self, encounter_ids):
        """
        Returns a new Dataset object that is a subset of the current dataset,
        containing only the specified encounter IDs.
        """
        subset = copy.deepcopy(self)
        subset.dataframe = subset.dataframe[self.dataframe[self.encounter_col].isin(encounter_ids)].copy()
        return subset
    
    def subsample_dataset(self, fraction):
        """
        Subsamples the dataset to the specified fraction of encounter IDs.
        """
        encounter_ids = self.encounter_ids()
        n_to_sample = int(len(encounter_ids) * fraction)
        
        # sample the encounter IDs without replacement and with a fixed random seed
        random_seed = self.seed
        sampled_encounter_ids = np.random.default_rng(seed=random_seed).choice(encounter_ids, size=n_to_sample, replace=False)
        self = self.create_subset(sampled_encounter_ids)
        return self