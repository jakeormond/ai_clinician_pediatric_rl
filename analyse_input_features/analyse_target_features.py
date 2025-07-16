import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import load_config


def basic_analysis(dataframe: pd.DataFrame, features_to_use: pd.DataFrame):
    """
    this is very basic analysis. Include things like number of encouters,
    encounter length, number of features, etc.    
    
    """

    n_encounters = dataframe['encounter_id'].nunique()
    print(f"Number of encounters: {n_encounters}")

    return


def vasoactive_inotropes_analysis(dataframe: pd.DataFrame, features_to_use: pd.DataFrame):
    # the vis features are part of the vasoactive_inotropic_infusion 
    # feature_set, with the vis feature being the aggregate feature. 
    # The features should be: dobutamine, dopamine, epinephrine, milrinone,
    # norepinephrine, vasopressin, vis.

    features = features_to_use[features_to_use['feature_set'] == 'vasoactive_inotropic_infusion']

    expected_feature_names = [
        'dobutamine', 'dopamine', 'epinephrine', 'milrinone',
        'norepinephrine', 'vasopressin', 'vis'
    ]

    assert set(features['feature']) == set(expected_feature_names), \
        f"Expected features {expected_feature_names}, but found {set(features['feature_name'])}"
     
    n_rows = dataframe.shape[0]

    # create a figure with subplots for each feature (we can do 2 rows of 4, and will 
    # add an extra plot for the clipped vis)

    expected_feature_names.extend(['vis_clipped', 'vis_transformed', 'vis_transformed_w_zeros'])

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, feature in enumerate(expected_feature_names):
        if feature in ['vis_clipped', 'vis_transformed', 'vis_transformed_w_zeros']:
            feature_data = dataframe["vis"].dropna()
        else:
            feature_data = dataframe[feature].dropna()

        n_feature_rows = feature_data.shape[0]
        print(f"Feature: {feature}, Rows with data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

        # drop all zeros from the feature data
        if feature != 'vis_transformed_w_zeros':
            feature_data = feature_data[feature_data != 0]
            n_feature_rows = feature_data.shape[0]
            print(f"Feature: {feature}, Rows with non-zero data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

        if feature == 'vis_clipped':
            # calculate the interquartile range (IQR) for the vis feature
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            iqr = q3 - q1

            # clip the data to 5*IQR above the 75th percentile
            upper_bound = q3 + 20 * iqr

            # print the upper bound and the number of values above it
            print(f"Upper bound for {feature}: {upper_bound}")
            print(f"Number of values above upper bound: {feature_data[feature_data > upper_bound].shape[0]}")

            # clip the data
            feature_data = feature_data.clip(upper=upper_bound)

            # print the min value
            min_value = feature_data.min()
            print(f"Min value for {feature}: {min_value}")

        elif feature in ['vis_transformed', 'vis_transformed_w_zeros']:

            # cubert transform the data
            # feature_data = np.power(feature_data, 1/5)  # add a small constant to avoid log(0)
           
            if feature == 'vis_transformed':
                pt = PowerTransformer(method='yeo-johnson')
                feature_data = pt.fit_transform(feature_data.values.reshape(-1, 1)).flatten()
            elif feature == 'vis_transformed_w_zeros':
                feature_data = pt.transform(feature_data.values.reshape(-1, 1)).flatten()                

        # create a sns.kde plot for the feature
        sns.kdeplot(feature_data, ax=axes[i], fill=True)
        axes[i].set_title(feature)
        
    plt.tight_layout()
    # plt.show()

    path = "plots/targets/vasoactive_inotropes_analysis.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)

    return


def iv_input_analysis(dataframe: pd.DataFrame, features_to_use: pd.DataFrame):
    """
    box_intake appears to do a nice job of normalizing the non-zero intake values,
    but is not very good at handling zeros. For that reason, we will initially go 
    with the cuberoot transformation, which seems to do reasonably well.
    """

    feature = "intake"
    feature_data = dataframe[feature]
    n_rows = len(feature_data)

    # drop nans
    feature_data = feature_data.dropna()
    n_feature_rows = len(feature_data)

    print(f"Feature: {feature}, Rows with data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(2):

        if i == 0:
            # drop all zeros from the feature data
            feature_data_temp = feature_data[feature_data != 0].copy()
            n_feature_rows = len(feature_data_temp)
            print(f"Feature: {feature}, Rows with non-zero data: {n_feature_rows} out of {n_rows} ({n_feature_rows / n_rows * 100:.2f}%)")

        else: # keep all data, including zeros
            feature_data_temp = feature_data.copy()
            
        sns.kdeplot(feature_data_temp, ax=axes[0 + i*5], fill=True)
        axes[0 + i*5].set_title(feature)

        min_value = feature_data_temp.min()
        print(f"Min value for {feature}: {min_value}")
        log_data = np.log(feature_data_temp + 1e-5)  # add a small constant to avoid log(0)
        sns.kdeplot(log_data, ax=axes[1 + i*5], fill=True)
        axes[1 + i*5].set_title(f"log_{feature}")

        sqrt_data = np.sqrt(feature_data_temp + 1e-5)  # add a small constant to avoid log(0)
        sns.kdeplot(sqrt_data, ax=axes[2 + i*5], fill=True)
        axes[2 + i*5].set_title(f"sqrt_{feature}")

        box_data = boxcox(feature_data_temp + 1e-5)  # add a small constant to avoid log(0)
        sns.kdeplot(box_data, ax=axes[3 + i*5], fill=True)
        axes[3 + i*5].set_title(f"box_{feature}")

        cuberoot_data = np.cbrt(feature_data_temp + 1e-5)  # add a small constant to avoid log(0)
        sns.kdeplot(cuberoot_data, ax=axes[4 + i*5], fill=True)
        axes[4 + i*5].set_title(f"cuberoot_{feature}")

    plt.tight_layout()
    # plt.show()

    path = "plots/targets/intake.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)

    return


def main():
    hospital = 'choc'

    data_config_path = f"config/{hospital}.yaml"
    data_config = load_config(data_config_path)

    features_to_use_path = data_config["features_to_use_path"]
    features_to_use = pd.read_csv(features_to_use_path)

    dataframe = pd.read_parquet(data_config["dataframe_path"])

    vasoactive_inotropes_analysis(dataframe, features_to_use)

    # iv_input_analysis(dataframe, features_to_use)

    pass



if __name__ == "__main__":
    main()
