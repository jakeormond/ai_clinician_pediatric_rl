import os
import pandas as pd
import copy
import numpy as np

import logging
logger = logging.getLogger(__name__)

# this class holds the complete dataset, the folds can 
# access the data they need using the encounter IDs
# and the methods in this class
# Note that rewards are loaded here, but processed in data_processor.py

class Dataset:
    def __init__(self, config, drop_actions=True, transform_target=True): # note that actions get dropped only for supervised learning, not RL
        
        # Dataset should store all the key value pairs from the config
        for key, value in config.items():
            setattr(self, key, value)
        
        self._load_dataset(drop_actions)
        
        if config['model_type'] == 'supervised_learning' and transform_target: # this is generally used for supervised learning, so gets
            self._transform_target() 


    def save_dataframe(self, config):

        loaded_df_path = config.data_config['dataframe_path']
        load_df_filename = os.path.basename(loaded_df_path)
        # strip .parquet from filename
        load_df_filename = load_df_filename.replace(".parquet", "")

        proportion_w_data = config.data_config['proportion_w_data']

        filename = f"{load_df_filename}_prop_{proportion_w_data}.parquet"
        path = config.get_experiments_dir + "/" + filename
        self.dataframe.to_parquet(path, index=False)
        logger.info(f"DataFrame saved to {path}")


    def _load_dataset(self, drop_actions):
        self.dataframe = pd.read_parquet(self.dataframe_path) # here I will have to add code to load the df with the reward
        
        # rename the encounter and time columns to standard names
        self.dataframe = self.dataframe.rename(columns={
            self.encounter_col: "encounter_id",
            self.time_col: "time"
        })
        self.encounter_col = "encounter_id"
        self.time_col = "time"
        
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

        ################# WE ACTUALLY NEED TO KEEP MORTALITY FOR ONLINE ANALYSIS DURING TRAINING!!!!!!!!!!!! ################
        # make sure to handle it correctly in data_processor.py
        
        if self.model_type == "reinforcement_learning":
            if self.use_terminal_reward:
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

        logger.info(f"Removed {n_encounter_ids_before - n_encounter_ids_after} "
                f"encounter IDs and {n_rows_before - n_rows_after} rows with NaN values.")
        logger.info(f"Remaining encounter IDs: {n_encounter_ids_after}, Remaining rows: {n_rows_after}")     

        if self.use_reward:
            self._load_reward()

        if self.truncate_trajectories:
            
            min_traj_len = self.min_traj_len
            max_traj_length = self.max_traj_length
            
            logger.info("Truncating trajectories..."
                  f" to a maximum length of {max_trajectory_length} and a minimum length of {min_traj_len}.")

            # identify encounter_ids with mortality == 1
            mortality_encounter_ids = self.dataframe[self.dataframe['mortality'] == 1][self.encounter_col].unique()
            
            # truncate each trajectory (identified by encounter_id) to max_trajectory_length
            df_limited = self.dataframe.groupby(self.encounter_col).head(max_trajectory_length).reset_index(drop=True)

            # get the lengths of each trajectory (identified by encounter_id)
            traj_lengths = df_limited.groupby(self.encounter_col).size()

            # remove any trajectories that are shorter than min_traj_len
            valid_encounter_ids = traj_lengths[traj_lengths >= min_traj_len].index
            df_limited = df_limited[df_limited[self.encounter_col].isin(valid_encounter_ids)].reset_index(drop=True)

            # for each mortality encounter_id, ensure that the last row has mortality == 1
            df_limited.loc[
                df_limited['encounter_id'].isin(mortality_encounter_ids) & 
                (df_limited.groupby('encounter_id')['time'].transform('max') == df_limited['time']),
                'mortality'
            ] = 1

            # logger.info how many encounter_ids are left, and how many mortality encounter_ids are left
            n_encounter_ids_limited = df_limited[self.encounter_col].nunique()
            n_mortality_encounter_ids_limited = df_limited[df_limited['mortality'] == 1][self.encounter_col].nunique()
            logger.info(f"After truncating, {n_encounter_ids_limited} encounter IDs remain, "
                  f"of which {n_mortality_encounter_ids_limited} have mortality == 1.")

            self.dataframe = df_limited.copy()


        return self
    

    def _load_reward(self):
        # load parquet file with rewards, path is self.reward_path
        reward_df = pd.read_parquet(self.reward_path)

        # confirm that reward_df and self.dataframe have the length
        if reward_df.shape[0] != self.dataframe.shape[0]:
            raise ValueError(f"Reward dataframe has {reward_df.shape[0]} rows, but main dataframe has {self.dataframe.shape[0]} rows.")
        
        # save the length of the dataframe before merging
        n_rows_before = self.dataframe.shape[0]

        # confirm that reward_df a "health_score" column
        if 'health_score' not in reward_df.columns:
            raise ValueError("Reward dataframe does not have a 'health_score' column.")
        
        # merge reward_df with self.dataframe on encounter_col and time_col
        self.dataframe = pd.merge(self.dataframe, reward_df[[self.encounter_col, self.time_col, 'health_score']],
                                  on=[self.encounter_col, self.time_col], how='inner')
        
        # confirm that the length of the dataframe is unchanged
        n_rows_after = self.dataframe.shape[0]
        if n_rows_before != n_rows_after:
            raise ValueError(f"After merging reward, dataframe has {n_rows_after} rows, but before merging had {n_rows_before} rows.")



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
        logger.info(f"Found {len(target_encounter_ids)} encounter IDs with target_data = 1")
        
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