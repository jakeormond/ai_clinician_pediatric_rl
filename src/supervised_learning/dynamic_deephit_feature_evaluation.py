"""for evaluating feature importance of the transformer
implementation of dynamic deephit"""

import data
import torch
import numpy as np
import os
import copy
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    all_features_attr = {
        'discharge': defaultdict(lambda: torch.zeros(n_features)),
        'death': defaultdict(lambda: torch.zeros(n_features))
    }
    window_length = config_manager.training_config['window_length']
    all_times_attr = {
        'discharge': defaultdict(lambda: torch.zeros(window_length)),
        'death': defaultdict(lambda: torch.zeros(window_length))
    }

    all_attr = {
        'discharge': defaultdict(lambda: torch.zeros(n_features, window_length)),
        'death': defaultdict(lambda: torch.zeros(n_features, window_length))
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
                global_temporal_imp = attr.abs().sum(dim=(2)).mean(dim=0).cpu().numpy()  # sum over batch and features
                global_imp = attr.abs().mean(dim=0).cpu().numpy()  # mean over batch

                # accumulate
                all_features_attr[event_type][timebin] += global_feat_imp
                all_times_attr[event_type][timebin] += global_temporal_imp
                all_attr[event_type][timebin] += global_imp.T

    for event_type in event_types:
        for timebin in range(n_timebins):
        # divide by number of samples to get mean
            all_features_attr[event_type][timebin] /= n_batches
            all_times_attr[event_type][timebin] /= n_batches
            all_attr[event_type][timebin] /= n_batches

    print("Completed attribution for cohort")        
    return all_features_attr, all_times_attr, all_attr

def get_feature_attributions(config_manager, n_batches=None):
    
    model_dir = get_model_dir(config_manager)
    save_dir = os.path.join(model_dir, 'attributions')
    # check if save_dir exists 
    if not os.path.exists(save_dir):
        hyperparams, time_bins = load_hyperparams(model_dir)    
        n_time_bins = len(time_bins) - 1
        
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='test')
        feature_names = test_processor.dynamic_deephit_dataset['feature_names']
        
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

        all_features_attr = {}
        all_times_attr = {}
        all_attr = {}
        for cohort_name, stay_ids in cohort_ids.items():
            print(f"Processing cohort: {cohort_name} with {len(stay_ids)} patients")
            
            # make a deep copy of the test_set
            cohort_processor = copy.deepcopy(test_processor)
            
            if cohort_name != 'all':   
                # filter the cohort_set to only include the stay_ids in this cohort
                cohort_processor.dynamic_deephit_dataset = filter_cohort(cohort_processor.dynamic_deephit_dataset, list(stay_ids))
            
            all_features_attr[cohort_name], all_times_attr[cohort_name], all_attr[cohort_name] = process_cohort_attributions(
                dynamic_deephit_model,
                cohort_processor,
                config_manager,
                batch_size=batch_size,
                n_batches=n_batches
            )

        # save the attributions
        os.makedirs(save_dir, exist_ok=True)
        for cohort_name, cohort_attr in all_features_attr.items():
            for event_type, event_attr in cohort_attr.items():
                
                data = {}
                for timebin, attr_values in event_attr.items():
                    data[f'timebin_{timebin}'] = attr_values
                
                attr_feature_df = pd.DataFrame({
                    'feature': test_processor.dynamic_deephit_dataset['feature_names'],
                    **data
                })
                
                filename = f"feature_attributions_{cohort_name}_{event_type}.csv"
                attr_feature_df.to_csv(os.path.join(save_dir, filename), index=False)
                print(f"Saved attributions to {filename}")

            for event_type, event_attr in all_times_attr[cohort_name].items():
                data = {}
                for timepoint, attr_values in event_attr.items():
                    data[f'timepoint_{timepoint}'] = attr_values
                
                attr_time_df = pd.DataFrame(data)
                
                filename = f"temporal_attributions_{cohort_name}_{event_type}.csv"
                attr_time_df.to_csv(os.path.join(save_dir, filename), index=False)
                print(f"Saved temporal attributions to {filename}")

            for event_type, event_attr in all_attr[cohort_name].items():
                # combine the arrays across timebins
                all_attr_array = np.stack([event_attr[timebin] for timebin in range(len(event_attr))], axis=1)  # [Features, Timebins]
                # save as npy
                filename = f"full_attributions_{cohort_name}_{event_type}.npy"
                np.save(os.path.join(save_dir, filename), all_attr_array)
                print(f"Saved full attributions to {filename}")

                pass # currently not saving the full attributions

    else: # load existing attributions
        print(f"Attributions already exist in {save_dir}, loading...")
        cohorts = [folder for folder in os.listdir(os.path.join(config_manager.paths['data_dir'], 'cohorts'))]
        cohorts = ['all'] + cohorts
        all_features_attr = {}
        all_times_attr = {}
        all_attr = {}
        for cohort_name in cohorts:
            all_features_attr[cohort_name] = {}
            all_times_attr[cohort_name] = {}
            all_attr[cohort_name] = {}
            for event_type in ['discharge', 'death']:
                filename = f"feature_attributions_{cohort_name}_{event_type}.csv"
                attr_features_df = pd.read_csv(os.path.join(save_dir, filename))
                
                event_features_attr = {}
                for timebin in range(len(attr_features_df.columns) - 1):
                    event_features_attr[timebin] = attr_features_df[f'timebin_{timebin}'].values
                all_features_attr[cohort_name][event_type] = event_features_attr

                filename = f"temporal_attributions_{cohort_name}_{event_type}.csv"
                attr_times_df = pd.read_csv(os.path.join(save_dir, filename))

                event_times_attr = {}
                for timepoint in range(len(attr_times_df.columns)):
                    event_times_attr[timepoint] = attr_times_df[f'timepoint_{timepoint}'].values
                all_times_attr[cohort_name][event_type] = event_times_attr        

                filename = f"full_attributions_{cohort_name}_{event_type}.npy"
                all_attr_array = np.load(os.path.join(save_dir, filename))
                all_attr[cohort_name][event_type] = {}
                for timebin in range(all_attr_array.shape[1]):
                    all_attr[cohort_name][event_type][timebin] = all_attr_array[:, timebin]

        feature_names = attr_features_df['feature'].values
                
        print("Loaded all attributions from disk")

    plot_feature_importance_by_time(all_attr, feature_names, save_dir)
    plot_feature_importance(all_features_attr, feature_names, save_dir)
    plot_temporal_importance(all_times_attr, save_dir)

    return all_features_attr, all_times_attr, all_attr, feature_names


def plot_feature_importance_by_time(all_attr, feature_names, save_path=None):
    # plot heatmaps of feature importance across timebins
    cohorts = list(all_attr.keys())
    event_types = list(all_attr[cohorts[0]].keys())
    timebins = list(all_attr[cohorts[0]][event_types[0]].keys())    
    n_timebins = len(timebins)
    for event_type in event_types:
        for cohort_name in cohorts:
            for timebin in range(n_timebins):
                attr = all_attr[cohort_name][event_type][timebin]
                # concatenate all timebins for plotting

                plt.figure(figsize=(18, 12))
                            
                norm = colors.PowerNorm(gamma=0.5, vmin=0, vmax=0.5)
                sns.heatmap(
                    attr, 
                    norm=norm, 
                    cmap='magma', 
                    xticklabels=10, 
                    yticklabels=feature_names,
                    cbar_kws={'label': 'Importance (Power Scaled: γ=0.5)'})

                plt.title(f"Feature Attribution across Prediction Horizons - Event: {event_type} - Cohort: {cohort_name} - Timebin: {timebin}")
                plt.xlabel("Features")
                plt.ylabel("Prediction Horizon (Bin)")
                plt.tight_layout(pad=2.0, w_pad=4.0, h_pad=2.0)
                plt.savefig(os.path.join(save_path, f'feature_importance_by_time_{cohort_name}_{event_type}_{timebin}.png'))
                plt.close()
                print(f"Saved feature importance by time plot to feature_importance_by_time_{cohort_name}_{event_type}_{timebin}.png")





def plot_temporal_importance(all_times_attr, save_path=None):
    n_timebins = len(all_times_attr['all']['death'].keys())
    event_types = ['discharge', 'death']
    cohorts = list(all_times_attr.keys())

    for event_type in event_types:
        plt.figure(figsize=(10, 6))
        for cohort_name in cohorts:
            attr = all_times_attr[cohort_name][event_type]
            # concatenate all timepoints for plotting
            data_array = np.array([attr[timepoint] for timepoint in range(n_timebins)])  # [Timepoints]

            plt.figure(figsize=(12, 5))
                        
            sns.heatmap(
                data_array, 
                norm=colors.PowerNorm(gamma=0.5), 
                cmap='magma', 
                xticklabels=10, 
                yticklabels=[f'Bin {i+1}' for i in range(8)],
                cbar_kws={'label': 'Importance (Power Scaled: γ=0.5)'})

            plt.title(f"Temporal Attribution Focus across Prediction Horizons - Event: {event_type} - Cohort: {cohort_name}")
            plt.xlabel("Window Timepoints (hours)")
            plt.ylabel("Prediction Horizon (Bin)")
            plt.tight_layout(pad=2.0, w_pad=4.0, h_pad=2.0)
            plt.savefig(os.path.join(save_path, f'temporal_importance_{cohort_name}_{event_type}.png'))
            plt.close()
            print(f"Saved temporal importance plot to temporal_importance_{cohort_name}_{event_type}.png")

def plot_feature_importance(all_attr, feature_names, normalize=False, save_path=None):
    n_timebins = len(all_attr['all']['death'].keys())
    event_types = ['discharge', 'death']
    cohorts = list(all_attr.keys())

    for event_type in event_types:
        df = pd.DataFrame(feature_names, columns=['feature'])
        mean_attr = {}
        for cohort_name in cohorts:
            attr = all_attr[cohort_name][event_type]
            # concatenate all timebins for plotting
            attr = np.array([attr[timebin] for timebin in range(n_timebins)])  # [Timebins, Features]
            # take the mean across timebins
            mean_attr[cohort_name] = attr.mean(axis=0)

            if normalize:
                # normalize to sum to 1
                mean_attr[cohort_name] = mean_attr[cohort_name] / mean_attr[cohort_name].sum()

            # add cohort column
            df[cohort_name] = mean_attr[cohort_name]

        for cohort_name, cohort_attr in all_attr.items():
            # create a new list of cohorts, where cohort_name is first, then the rest
            other_cohorts = [c for c in cohorts if c != cohort_name]
            ordered_cohorts = [cohort_name] + other_cohorts
            
            # order the dataframe columns accordingly
            df_ordered = df[['feature'] + ordered_cohorts]
            # order the rows by the values of the first cohort
            df_ordered = df_ordered.sort_values(by=cohort_name, ascending=False)

            # plot the top 20 features as horizontal bar plot
            plt.figure(figsize=(10, 6))     
            sns.barplot(data=df_ordered.head(20).melt(id_vars='feature', value_vars=ordered_cohorts), 
                        y='feature', x='value', hue='variable', orient='h')
            plt.xlabel('Mean Absolute Attribution')
            plt.ylabel('Features')
            plt.title(f'Feature Importance for {cohort_name} - Event: {event_type}')   
                
            plt.tight_layout(pad=2.0, w_pad=4.0, h_pad=2.0)
            if normalize:
                plot_filename = f'feature_importance_normalized_{cohort_name}_{event_type}.png'
            else:
                plot_filename = f'feature_importance_{cohort_name}_{event_type}.png'
            plt.savefig(os.path.join(save_path, plot_filename))
            plt.close()
            print(f"Saved feature importance plot to {plot_filename}")



        pass




def main():
    pass    

if __name__ == "__main__":
    main()