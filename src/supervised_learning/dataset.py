import pandas as pd

# this class holds the complete dataset, the folds can 
# access the data they need using the encounter IDs
# and the methods in this class
class Dataset:
    def __init__(self, config):
        
        self.target_col = config["target_col"]
        self.encounter_col = config["encounter_col"]
        self.time_col = config["time_col"]
        self.config = config
        
        self._load_dataset()
        self._transform_target()

    def _load_dataset(self):
        self.dataframe = pd.read_parquet(self.config["dataframe_path"])
        features_to_use_df = pd.read_csv(self.config["features_to_use_path"])

         # drop any rows where "do not use" is 1
        features_to_use_df = features_to_use_df[features_to_use_df['do not use'] != 1]

        # drop any rows where "proportion_sepsis" is less than proportion_w_data and
        # imputation_method is "interpolation"
        features_to_use_df = features_to_use_df[
            (features_to_use_df['proportion_sepsis'] >= self.config["proportion_w_data"]) |   
            (features_to_use_df['imputation_method'] != 'interpolation')
        ]

        # drop all features in the action_feature_sets
        for feature_set in self.config["action_feature_sets"]:
            features_to_use_df = features_to_use_df[features_to_use_df['feature_set'] != feature_set]

        # make list of features
        features = features_to_use_df['feature'].tolist()

        # add to these the encounter_col, time_col,
        features_to_keep = [self.config["encounter_col"], self.config["time_col"]] + features

        self.dataframe = self.dataframe[features_to_keep].copy()
        
        n_encounter_ids_before = self.dataframe[self.config["encounter_col"]].nunique()
        n_rows_before = self.dataframe.shape[0]  

        # remove rows with NaN values anywhere in the dataframe
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe.reset_index(drop=True)

        n_encounter_ids_after = self.dataframe[self.config["encounter_col"]].nunique()
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

        encounter_ids = self.dataframe[self.config["encounter_col"]].unique()

        # find the encounter_ids that have target_data equal to 1
        target_encounter_ids = self.dataframe[self.dataframe[self.target_col] == 1][self.encounter_col].unique()
        print(f"Found {len(target_encounter_ids)} encounter IDs with target_data = 1")
        
        # get the surviving encounter_ids
        surviving_encounter_ids = set(encounter_ids) - set(target_encounter_ids)
        
        if self.config["prediction_type"] == "mortality_all":
            # all values for the target_data are 1 for the encounter_ids that have target_data equal to 1
            self.dataframe[self.target_col] = self.dataframe[self.encounter_col].apply(
                lambda x: 1 if x in target_encounter_ids else 0
            )       

        else:
            non_sur_data = self.dataframe[
                    self.dataframe[self.encounter_col].isin(target_encounter_ids)
                ].copy()
            
            if self.config["prediction_type"] == "mortality_24":
                # just the last 24 rows are labelled as 1 for the encounter_ids that have target_data equal to 1
                # Get data for encounters that should have mortality=1
                

                last_24_rows = non_sur_data.groupby(self.encounter_col).tail(24)

                # Set target to 1 for these rows
                self.dataframe.loc[last_24_rows.index, self.target_col] = 1  

            elif self.config["prediction_type"] == "discharge_mortality":                    
                
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

    def modeling_data(self, encounter_ids):
        """
        Returns the modeling data as a DataFrame.
        This is the data that will be used for training and testing.
        """
        
        modeling_data = self.dataframe[self.dataframe[self.encounter_col].isin(encounter_ids)].copy()
        
        encounters_and_times = modeling_data[[self.encounter_col, self.time_col]]
        
        X_train = modeling_data.drop(columns=[self.encounter_col, self.time_col, self.target_col])
        y_train = modeling_data[self.target_col]
                
        return X_train, y_train, encounters_and_times   
    
    def encounter_ids(self):
        """
        Returns the unique encounter IDs in the dataset.
        """
        return self.dataframe[self.encounter_col].unique()