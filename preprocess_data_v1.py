import yaml
import pandas as pd
import pyarrow


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def load_dataset(config, proportion_w_data=0.8):

    dataframe = pd.read_parquet(config["DATAFRAME_PATH"])

    features_to_use_df = pd.read_csv(config["FEATURES_TO_USE_PATH"])
    
    # drop any rows where "do not use" is 1
    features_to_use_df = features_to_use_df[features_to_use_df['do not use'] != 1]

    # drop any rows where "propition_sepsis" is less than proportion_w_data and
    # imputation_method is "interpolation"
    features_to_use_df = features_to_use_df[
        (features_to_use_df['proportion_sepsis'] >= proportion_w_data) |   
        (features_to_use_df['imputation_method'] != 'interpolation')
    ]

    features_to_keep = features_to_use_df['feature'].tolist()
    # add to these the ENCOUNTER_COL, TIME_COL, and the REWARD
    features_to_keep = [config["ENCOUNTER_COL"], config["TIME_COL"], \
                                    config["REWARD"]] + features_to_keep

    dataframe = dataframe[features_to_keep].copy()
    
    n_encounter_ids_before = dataframe[config["ENCOUNTER_COL"]].nunique()
    n_rows_before = dataframe.shape[0]  

    # remove rows with NaN values anywhere in the dataframe
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop=True)

    n_encounter_ids_after = dataframe[config["ENCOUNTER_COL"]].nunique()
    n_rows_after = dataframe.shape[0]

    print(f"Removed {n_encounter_ids_before - n_encounter_ids_after} encounter IDs and {n_rows_before - n_rows_after} rows with NaN values.")
    print(f"Remaining encounter IDs: {n_encounter_ids_after}, Remaining rows: {n_rows_after}")


    return dataframe, features_to_use_df


def split_train_test(dataframe, include_val=False, val_size=0., test_size=0.2):

    n_encounters = dataframe[config["ENCOUNTER_COL"]].nunique()





   
def split_into_rl_dataset(dataframe, config):

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

    rewards = dataframe[config["REWARD"]].copy()    
    actions = dataframe[config["TARGETS"]].copy()
    observations = dataframe.drop(columns=[config["REWARD"]] + config["TARGETS"]).copy()

    # compute terminated
    # terminate is 0 everywhere except the final timestep for each patient
    dataframe['terminated'] = 0
    dataframe.loc[dataframe.groupby(config["ENCOUNTER_COL"]).tail(1).index, 'terminated'] = 1
    terminated = dataframe['terminated'].copy()

    # get the next observations
    next_observations = observations.copy()

    # get the indices of all the trajectory starst, i.e. the first index of each patient
    trajectory_starts = dataframe.groupby(config["ENCOUNTER_COL"]).head(1).index

    # get the trajectory ends, i.e. the last index of each patient
    trajectory_ends = dataframe.groupby(config["ENCOUNTER_COL"]).tail(1).index

    # for observations and actions, we remove the trajectory ends
    observations = observations.drop(index=trajectory_ends)
    actions = actions.drop(index=trajectory_ends)

    # for rewards, next_observations, and terminated, we remove the trajectory starts
    rewards = rewards.drop(index=trajectory_starts)
    next_observations = next_observations.drop(index=trajectory_starts)
    terminated = terminated.drop(index=trajectory_starts)

    # remove the encounter and time columns from observations and next_observations 
    observations = observations.drop(columns=[config["ENCOUNTER_COL"], config["TIME_COL"]])
    next_observations = next_observations.drop(columns=[config["ENCOUNTER_COL"], config["TIME_COL"]])

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


def main():
    hospital = 'choc'  # or 'mimic'
    config_path = f"config/{hospital}.yaml"
    config = load_config(config_path)

    dataframe, features_to_use_df = load_dataset(config, proportion_w_data=0.8)

    # create train and test, or train, test and validation splits
    train_data, test_data = split_train_test(dataframe, inlcude_val = False)

    
    
    dataset = split_into_rl_dataset(dataframe, config)
    
    pass

if __name__ == "__main__":
    main()