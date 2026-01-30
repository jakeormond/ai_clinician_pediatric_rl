"""for evaluating feature importance of the transformer
implementation of dynamic deephit"""

import torch
import numpy as np
import os
import copy
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from captum.attr import IntegratedGradients
from .training import create_buffer, create_processor_from_shards
from .predict import instantiate_model, filter_cohort


def get_top_features(attr_matrix, feature_names, cohort_mask):
    # Slice the attributions for the specific cohort
    cohort_attr = attr_matrix[cohort_mask] # [N_subset, Time, Features]
    
    # 1. Absolute mean across patients and time
    # This tells us which features MOVED the needle most, regardless of direction
    importance = np.abs(cohort_attr).mean(axis=(0, 1))
    
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)






def get_model_dir(config_manager):
    parent_dir = os.path.join(
        config_manager.paths['final_model_dir'],
        'transformer',
        'all',
        'final_model'
    )

    model_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    if len(model_folders) > 1:
        raise ValueError("Multiple model folders found in final_model_dir, please specify one")
    model_dir = os.path.join(parent_dir, model_folders[0])
    return model_dir

def load_hyperparams(model_dir):
    results_json_path = os.path.join(model_dir, 'results.json')
    with open(results_json_path, 'r') as f:
        results_dict = json.load(f)
    hyperparams = results_dict['best_hyperparameters']    
    time_bins = results_dict['time_bins']
    
    return hyperparams, time_bins


def get_cohort_ids(test_set, config_manager):
    # load the cohort ids
    cohorts = [folder for folder in os.listdir(os.path.join(config_manager.paths['data_dir'], 'cohorts'))]
    cohort_ids = {}
    cohort_ids['all'] = set(test_set['stay_ids'].tolist())
    for cohort in cohorts:
        cohort_file = os.path.join(config_manager.paths['data_dir'], 'cohorts', cohort, 'stay_ids.csv')
        df = pd.read_csv(cohort_file)
        cohort_ids_temp = set(df['stay_id'].tolist())
        # keep only those in the test set
        cohort_ids[cohort] = cohort_ids_temp.intersection(cohort_ids['all'])

    return cohort_ids



def process_cohort_attributions(
    model,
    cohort_processor,
    config_manager,
    batch_size=16,
    n_batches = 100
):
    # forward wrapper for captum
    def captum_forward_wrapper(windows_chunk, event_type, time_horizon_bin):
        """
        Captum passes a chunk of windows: (Batch, Window_Length, Features)
        We must return a single scalar per window: (Batch,)
        """
        # model(windows_chunk) returns [logits_event0, logits_event1]
        risk_logits_tuple, _ = model(windows_chunk)
        
        # Let's focus on Death (Event 0)
        event_logits = risk_logits_tuple[event_type] # (Batch, Window_Length, Num_Bins)
               
        # We explain the very last prediction in the window (the current risk)
        # at the furthest time horizon (Bin 7)
        target_output = event_logits[:, -1, time_horizon_bin]
        
        # Note: If your model requires softmax/cif conversion, do it here
        # return torch.softmax(target_output, dim=-1)
        return target_output

    cohort_buffer, _ = create_buffer(cohort_processor, batch_size, config_manager, shuffle=True)

    n_timebins = len(config_manager.time_bins) - 1
    event_types = ['discharge', 'death']  # Death and Discharge
    n_features = len(cohort_processor.dynamic_deephit_dataset['feature_names'])
    
    ig = IntegratedGradients(captum_forward_wrapper)
    all_attr = {
        'discharge': defaultdict(lambda: torch.zeros(n_features)),
        'death': defaultdict(lambda: torch.zeros(n_features))
    }

    for i, batch in enumerate(cohort_buffer):
        if i >= n_batches:
            break

        X = batch['features'].to('cuda')
        
        # Calculate attributions
        # n_steps=50 is standard; internal_batch_size helps GPU memory
        for timebin in range(n_timebins):
            for event_ind, event_type in enumerate(event_types):
                attr = ig.attribute(
                    X,
                    additional_forward_args=(event_ind, timebin),
                    n_steps=50, 
                    internal_batch_size=batch_size
                )

                # aggregate immediately to save memory
                global_feat_imp = attr.abs().sum(dim=(1)).mean(dim=0).cpu().numpy()  # sum over batch and time

                # accumulate
                all_attr[event_type][timebin] += global_feat_imp

    for event_type in event_types:
        for timebin in range(n_timebins):
        # divide by number of samples to get mean
            all_attr[event_type][timebin] /= n_batches

    print("Completed attribution for cohort")        
    return all_attr


def get_feature_attributions(config_manager, n_batches=None):
    
    model_dir = get_model_dir(config_manager)
    save_dir = os.path.join(model_dir, 'feature_attributions')
    # check if save_dir exists 
    if not os.path.exists(save_dir):
        hyperparams, time_bins = load_hyperparams(model_dir)    
        n_time_bins = len(time_bins) - 1
        
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='test')
        
        # set smaller batch size for attribution computation
        batch_size = 16

        # instantiate the model
        dynamic_deephit_model = instantiate_model(
            config_manager, 
            hyperparams, 
            n_time_bins, 
            num_risks=2, 
            test_processor=test_processor,
            experiment_dir=model_dir
        )
        dynamic_deephit_model.eval()
        
        # extract the test set
        test_set = test_processor.dynamic_deephit_dataset
    
        # load the cohort ids
        cohort_ids = get_cohort_ids(test_set, config_manager)

        all_attr = {}
        for cohort_name, stay_ids in cohort_ids.items():
            print(f"Processing cohort: {cohort_name} with {len(stay_ids)} patients")
            
            # make a deep copy of the test_set
            cohort_processor = copy.deepcopy(test_processor)
            
            if cohort_name != 'all':   
                # filter the cohort_set to only include the stay_ids in this cohort
                cohort_processor.dynamic_deephit_dataset = filter_cohort(cohort_processor.dynamic_deephit_dataset, list(stay_ids))
            
            all_attr[cohort_name] = process_cohort_attributions(
                dynamic_deephit_model,
                cohort_processor,
                config_manager,
                batch_size=batch_size,
                n_batches=n_batches
            )

        # save the attributions
        save_dir = os.path.join(model_dir, 'feature_attributions')
        os.makedirs(save_dir, exist_ok=True)
        for cohort_name, cohort_attr in all_attr.items():
            for event_type, event_attr in cohort_attr.items():
                
                data = {}
                for timebin, attr_values in event_attr.items():
                    data[f'timebin_{timebin}'] = attr_values
                
                attr_df = pd.DataFrame({
                    'feature': test_processor.dynamic_deephit_dataset['feature_names'],
                    **data
                })
                
                filename = f"attributions_{cohort_name}_{event_type}.csv"
                attr_df.to_csv(os.path.join(save_dir, filename), index=False)
                print(f"Saved attributions to {filename}")

    else: # load existing attributions
        print(f"Attributions already exist in {save_dir}, loading...")
        cohorts = [folder for folder in os.listdir(os.path.join(config_manager.paths['data_dir'], 'cohorts'))]
        cohorts = ['all'] + cohorts
        all_attr = {}
        for cohort_name in cohorts:
            all_attr[cohort_name] = {}
            for event_type in ['discharge', 'death']:
                filename = f"attributions_{cohort_name}_{event_type}.csv"
                attr_df = pd.read_csv(os.path.join(save_dir, filename))
                
                event_attr = {}
                for timebin in range(len(attr_df.columns) - 1):
                    event_attr[timebin] = attr_df[f'timebin_{timebin}'].values
                all_attr[cohort_name][event_type] = event_attr
        feature_names = attr_df['feature'].values
        print("Loaded all attributions from disk")

    return all_attr, feature_names


def plot_feature_importance(all_attr, feature_names, save_path):
    n_timebins = len(next(iter(all_attr['all']['death'].keys())))
    event_types = ['discharge', 'death']
    
    for cohort_name, cohort_attr in all_attr.items():
        for event_type in event_types:
            plt.figure(figsize=(15, 5 * n_timebins))
            for timebin in range(n_timebins):
                plt.subplot(n_timebins, 1, timebin + 1)
                attr_values = cohort_attr[event_type][timebin]
                
                # Create a DataFrame for plotting
                df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': attr_values
                }).sort_values(by='importance', ascending=False).head(20)
                
                sns.barplot(x='importance', y='feature', data=df, palette='viridis')
                plt.title(f'Cohort: {cohort_name} | Event: {event_type} | Timebin: {timebin}')
                plt.xlabel('Mean Absolute Attribution')
                plt.ylabel('Feature')
            
            plt.tight_layout()
            plot_filename = f'feature_importance_{cohort_name}_{event_type}.png'
            plt.savefig(os.path.join(save_path, plot_filename))
            plt.close()
            print(f"Saved feature importance plot to {plot_filename}")




def main():
    pass    

if __name__ == "__main__":
    main()