import argparse
import os
import sys
import json
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..config_manager import ConfigManager
from ..supervised_learning.dataset import Dataset
# from utilities import load_config


def parse_arguments():
    """
    we need to recreate the RL args in order to load the data as it 
    is done in the RL training script.
    """
    parser = argparse.ArgumentParser(description='checking data')
    parser.add_argument('--hospital', 
                        default='choc', 
                        choices=['choc', 'mimic'], 
                        help='Hospital dataset to use (default: choc)')
    
    parser.add_argument('--experiment_type', 
                        default='reinforcement_learning')
    
    parser.add_argument('--model_type', 
                        default='no_model') 
    
    parser.add_argument('--mode', type=str, default='analyse_features')

    parser.add_argument('--start_time', type=str, default = None,                  
                    help='Start time for this experiment run')
    
    parser.add_argument('--experiment_variant', 
                        default=None)

    parser.add_argument('--search_variant', 
                        default=None)
    
    args = parser.parse_args()

    # If start_time is not provided, generate a timestamp
    if args.start_time is None:
        args.start_time = time.strftime("%Y%m%d_%H%M%S")

    return args


def basic_analysis(dataframe: pd.DataFrame, features_to_use: pd.DataFrame):
    """
    this is very basic analysis. Include things like number of encouters,
    encounter length, number of features, etc.    
    
    """

    n_encounters = dataframe['encounter_id'].nunique()
    print(f"Number of encounters: {n_encounters}")

    return


def feature_analysis(dataframe: pd.DataFrame, feature, transform: str = "yeo-johnson"):
    # the vis features are part of the vasoactive_inotropic_infusion 
    # feature_set, with the vis feature being the aggregate feature. 
    # The features should be: dobutamine, dopamine, epinephrine, milrinone,
    # norepinephrine, vasopressin, vis.

    # vis_features = [
    #    'dobutamine', 'dopamine', 'epinephrine', 'milrinone',
    #    'norepinephrine', 'vasopressin', 'vis'
    # ]
    
    feature_data = dataframe[feature]
    
    # crate a figure with 2 rows of 2 subplots each
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(2): # run twice, once with zeros and once without

        if i == 1:
            # drop all zeros from the feature data
            feature_data_temp = feature_data[feature_data != 0].copy()
            n_feature_rows = len(feature_data_temp)
            n_rows = len(feature_data)
            print(f"Feature: {feature}, Rows with non-zero data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

        else: # keep all data, including zeros
            feature_data_temp = feature_data.copy()            

        sns.kdeplot(feature_data_temp, ax=axes[0 + i*2], fill=True)
        axes[0 + i*2].set_title(f'{feature}')

        if transform == "yeo-johnson":
            pt = PowerTransformer(method='yeo-johnson', standardize=False)    
            pt.fit(feature_data_temp.values.reshape(-1, 1))
            transformed_feature = pt.transform(feature_data_temp.values.reshape(-1, 1)).flatten()

        elif transform == 'log1p':
            transformed_feature = np.log1p(feature_data_temp)
        
        sns.kdeplot(transformed_feature, ax=axes[1 + i*2], fill=True)
        axes[1 + i*2].set_title(f'transformed_{feature} ({transform})')
    
    plt.tight_layout()
    # plt.show()

    path = f"plots/targets/{feature}_{transform}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)

    return


def main():
    args = parse_arguments()
    
    config_manager = ConfigManager(args)
 
    dataset = Dataset(config_manager.data_config)
    print(f"Dataset loaded with {len(dataset.dataframe)} rows and {len(dataset.dataframe.columns)} columns")

    features = ['vis', 'intake']
    transforms = ['yeo-johnson', 'log1p']

    for feature in features:
        for transform in transforms:
            feature_analysis(dataset.dataframe, feature, transform)

    pass



if __name__ == "__main__":
    main()
