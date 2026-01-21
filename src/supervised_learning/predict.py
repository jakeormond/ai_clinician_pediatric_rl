"""
For generating predictions using a trained Dynamic-DeepHit model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.

"""

# predict.py
import argparse
import json
import torch
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from torch.utils.data import DataLoader

from ..data.data_processor_for_shards import ShardProcessor
from ..rl.replay_buffer import DynamicDeepHit_Lazy, worker_init_fn
from ..supervised_learning.models.dynamic_deephit import DynamicDeepHitMulti
from ..supervised_learning.models.dynamic_deephit_transformer import TransformerDeepHitMulti
from ..supervised_learning.training import create_processor_from_shards
from ..supervised_learning.dynamic_deephit_evaluation import calculate_c_index, calculate_brier_score, calculate_roc_and_prc, calculate_all_bin_ece


def filter_cohort(dynamic_deephit_dataset, cohort_stay_ids):
    # filter the dynamic_deephit_dataset
    stay_ids = dynamic_deephit_dataset['stay_ids']
    print(f"Original number of stays in dynamic_deephit_dataset: {len(stay_ids)}")
    print(f"Original number of timesteps in dynamic_deephit_dataset: {dynamic_deephit_dataset['trajectories'].shape[0]}")

    # Create boolean mask for stays to keep
    mask = np.isin(stay_ids, cohort_stay_ids)
    indices_to_keep = np.where(mask)[0]
    print(f"Number of stays to keep for cohort: {len(indices_to_keep)}")

    # Build trajectory mask
    traj_mask = np.zeros(dynamic_deephit_dataset['trajectories'].shape[0], dtype=bool)
    for idx in indices_to_keep:
        start_idx = dynamic_deephit_dataset['index_ptr'][idx]
        end_idx = dynamic_deephit_dataset['index_ptr'][idx + 1]
        traj_mask[start_idx:end_idx] = True

    # Apply masks to all arrays at once
    dynamic_deephit_dataset['stay_ids'] = dynamic_deephit_dataset['stay_ids'][mask]
    dynamic_deephit_dataset['event'] = dynamic_deephit_dataset['event'][mask]
    dynamic_deephit_dataset['trajectories'] = dynamic_deephit_dataset['trajectories'][traj_mask]
    dynamic_deephit_dataset['predictions'] = dynamic_deephit_dataset['predictions'][traj_mask]
    dynamic_deephit_dataset['time_to_event_raw'] = dynamic_deephit_dataset['time_to_event_raw'][traj_mask]
    dynamic_deephit_dataset['time_to_event'] = dynamic_deephit_dataset['time_to_event'][traj_mask]

    # Rebuild index_ptr
    new_index_ptr = np.zeros(len(indices_to_keep) + 1, dtype=np.int32)
    for i, idx in enumerate(indices_to_keep):
        start_idx = dynamic_deephit_dataset['index_ptr'][idx]
        end_idx = dynamic_deephit_dataset['index_ptr'][idx + 1]
        new_index_ptr[i + 1] = new_index_ptr[i] + (end_idx - start_idx)
    dynamic_deephit_dataset['index_ptr'] = new_index_ptr

    return dynamic_deephit_dataset  


def instantiate_model(
        config_manager, 
        hyperparams, 
        n_time_bins, 
        num_risks, 
        model_type=None, 
        test_processor=None,
        experiment_dir=None
    ):
    
    feature_dim = test_processor.feature_dim
    if model_type == "GRU": # GRU-based Dynamic-DeepHit
        dynamic_deephit_model = DynamicDeepHitMulti(
            test_processor.feature_dim,
            hyperparams['hidden_dim'],
            hyperparams['attn_hidden_dim'],
            hyperparams['n_layers'],
            n_time_bins,
            num_risks,
            dropout=hyperparams['dropout'],
            use_layernorm=hyperparams['use_layernorm'],
            rnn_type="GRU",
            predict_covariates=test_processor.physiological_feature_ind,
        ).to(config_manager.training_config['device'])

    else: # currently only the separate transformer-based Dynamic-DeepHit is supported
        dynamic_deephit_model = TransformerDeepHitMulti(
            feature_dim=feature_dim,   # input feature dimension
            d_model=hyperparams['d_model'],               # e.g., 128 or 256
            nhead=hyperparams['nhead'],                   # e.g., 4 or 8
            num_encoder_layers=hyperparams['n_layers'],   # e.g., 3 or 6
            num_time_bins=n_time_bins,
            num_risks=num_risks,
            dropout=hyperparams['dropout'],
            ff_factor=hyperparams['ff_factor'],
            max_seq_len=config_manager.training_config['window_length'],
            predict_covariates=test_processor.physiological_feature_ind,       
        ).to(config_manager.training_config['device'])
    
    # load trained model weights from final_model.pt
    model_path = os.path.join(experiment_dir, 'final_model.pt')
    device = config_manager.training_config['device']
    dynamic_deephit_model.load_state_dict(torch.load(
        model_path, 
        map_location=device,
        weights_only=True
    ))
    dynamic_deephit_model.eval()   
    return dynamic_deephit_model

def inference_loop(
        model, 
        processor, 
        config_manager, 
        filename=None, 
        model_type=None, 
        experiment_dir=None,
        device='cuda'
        ):
    
    stay_ids = processor.dynamic_deephit_dataset['stay_ids']
    index_ptr = processor.dynamic_deephit_dataset['index_ptr']
    window_length = config_manager.training_config['window_length']

    # create storage for predictions
    all_predictions = []    
    model.eval()
    with torch.no_grad():
        for i, stay_id in enumerate(stay_ids):
            start_idx = index_ptr[i]
            end_idx = index_ptr[i+1]
            
            # 1. Get full patient data (T, F)
            trajectory = processor.dynamic_deephit_dataset['trajectories'][start_idx:end_idx]
            trajectory = torch.as_tensor(trajectory, dtype=torch.float32, device=device)
            T, F = trajectory.shape

            # 2. Add padding so we can predict at Hour 1
            # (window_length - 1 repeats of first step + actual trajectory)   
            pad = trajectory[0:1].repeat(window_length - 1, 1)  # (window_length - 1, F)
            padded_traj = torch.cat([pad, trajectory], dim=0)

            # 3. Create windows (T, window_length, F)
            # unfold(dimension, size, step)
            windows = padded_traj.unfold(0, window_length, 1).transpose(1, 2)
            
            # initialize loss mask (not used during inference, but required by model forward), zeros of shape (batch_size, window_length)
            # loss_mask = torch.ones((windows.shape[0], window_length), dtype=torch.float32, device=device)
            loss_mask = torch.zeros((windows.shape[0], window_length), dtype=torch.bool, device=device)

            n_padded_windows = min([window_length - 1, T])
            for t in range(n_padded_windows):
                loss_mask[t, :(window_length - (t + 1))] = 0

            # 4. Batch Inference
            batch_size = 64
            all_preds = []
            
            for chunk_start in range(0, T, batch_size):
                chunk_end = min(chunk_start + batch_size, T)
                windows_chunk = windows[chunk_start:chunk_end].to(device)
                mask_chunk = loss_mask[chunk_start:chunk_end].to(device)
                # make mask chunk boolean
                mask_chunk = mask_chunk.bool()

                if model_type == 'GRU':
                    risk_logits, _, _ = model(windows_chunk, mask=mask_chunk)
                    risk_logits = risk_logits.cpu()  # (batch_size, window_length, num_risks, num_time_bins)
                else:
                    # risk_logits, _ = model(windows_chunk, padding_mask=mask_chunk)
                    risk_logits, _ = model(windows_chunk)
                    risk_logits = [logits[:, -1, :].cpu() for logits in risk_logits]  # unpack tuple
                    risk_logits = torch.stack(risk_logits, dim=2)  # (batch_size, num_risks, num_time_bins)

                    # check if risk_logits has any nulls
                    if torch.isnan(risk_logits).any():
                        raise ValueError(f"NaN values found in risk_logits for stay_id {stay_id} at chunk {chunk_start}-{chunk_end}")

                # reshape to (T, K, D)
                final_step_logits = risk_logits.permute(0, 2, 1)

                # add to list of all predictions    
                all_preds.append(final_step_logits)
            
            # 5. concatenate all predictions for this patient
            patient_preds = torch.cat(all_preds, dim=0)  
            all_predictions.append(patient_preds)

    # concatenate all patient predictions into a numpy array
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()

    # add predictions to processor
    processor.dynamic_deephit_dataset['predictions'] = all_predictions

    # save all values in processor.dynamic_deephit_dataset to an h5 file
    if filename is not None:
        file_path = os.path.join(experiment_dir, filename + '.h5')
    else:
        file_path = os.path.join(experiment_dir, 'test_set_predictions.h5')
    
    processor.dynamic_deephit_dataset_to_h5(file_path)
    dynamic_deephit_dataset = processor.dynamic_deephit_dataset

    return dynamic_deephit_dataset


def predict(config_manager):

    filename = 'test_set_predictions'
    filepath = os.path.join(config_manager.experiment_dir, filename + '.h5')
    if os.path.exists(filepath):
        # load existing predictions
        print(f"Loading existing predictions from {filepath}")
        dynamic_deephit_dataset = ShardProcessor.load_dynamic_deephit_dataset_from_h5(filepath, config_manager)

    else:
        # load result.json from config_manager.experiment_dir to get hyperparameters
        with open(
            os.path.join(config_manager.experiment_dir, 'results.json'), 'r'
        ) as f:
            results_dict = json.load(f)
        hyperparams = results_dict['best_hyperparameters']    
        time_bins = results_dict['time_bins']
        n_time_bins = len(time_bins) - 1
        og_config_manager = results_dict['config']
        
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        _, _, test_processor = create_processor_from_shards(config_manager, normalize=True)
        
        # instantiate the model
        dynamic_deephit_model = instantiate_model(config_manager, hyperparams, n_time_bins, num_risks=2, test_processor=test_processor)

        # run inference on test set
        dynamic_deephit_dataset = inference_loop(dynamic_deephit_model, test_processor, config_manager,device='cuda')


def plot_c_index(c_index_values, statistics, plot_dir):
    # create 2 subplots, one for each model type, and plot all the pairs of c-index values as boxplots
    models = c_index_values.keys()
    cohorts = c_index_values['GRU'].keys()
    fig, axes = plt.subplots(2, len(cohorts), figsize=(18, 8), sharey=True)
    
    for i, model_type in enumerate(models):
        for j, cohort in enumerate(cohorts):
            data = []
            
            model_types = ['all', cohort]
            # plot boxplots for each model type
            
            for model_cohort in model_types:
                for c_index in c_index_values[model_type][cohort][model_cohort]:
                    data.append({
                        'C-index': c_index, 
                        'Cohort': cohort,
                        'Model Type': model_cohort
                    })

            # get p-value for this cohort
            p_value = statistics[model_type][cohort]['p_value']   

            # plot boxplot at subplot axes[i, j-1]       
            ax = axes[i, j] 
                           
            df = pd.DataFrame(data)
            sns.boxplot(
                data=df,
                x='Cohort',
                y='C-index',
                hue='Model Type',
                ax=ax
            )

            # plot significance star
            # if p_value < 0.001:
            #     ax.text(0.5, max(df['C-index']) * 0.95, '**', ha='center', va='bottom', fontsize=20)
            # elif p_value < 0.05:
            #     ax.text(0.5, max(df['C-index']) * 0.95, '*', ha='center', va='bottom', fontsize=20)
            
            if model_type == 'transformer':
                model_title = 'TF'
            else:
                model_title = model_type.upper()
            ax.set_title(f"{model_title} Model - Cohort: {cohort}\n(p-value: {p_value:.4f})")
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'c_index_comparison.png')
    plt.savefig(plot_path)
    print(f"C-index comparison plot saved to {plot_path}")
            


def predict_general_models(config_manager):
    models = ['GRU', 'transformer']
    cohort = 'all'
    config_manager.reset_cohort(cohort)         
    c_index_values = {}

    for model_type in models:
        c_index_values[model_type] = []






def predict_all(config_manager): # for dynamic deephit models only
    models = ['GRU', 'transformer']
    # cohorts = ['all', 'sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad']
    cohorts = ['sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad']

    c_index_values = {}
    statistics = {}

    for model_type in models:
        c_index_values[model_type] = {}
        statistics[model_type] = {}

        # we need to calculate c-index for each cohort using either the general or cohort-specific model
        # and compare
        for cohort in cohorts:
            if cohort == 'all':
                continue

            model_types = ['all', cohort]
            c_index_values[model_type][cohort] = {}
            statistics[model_type][cohort] = {}

            model_types_reversed = model_types[::-1]
            
            # create storage for logits, time_to_event, event_label
            logits_dict = {}
            time_to_event_dict = {}
            event_label_dict = {}
            
            for i, model_cohort in enumerate(model_types_reversed):
                c_index_values[model_type][cohort][model_cohort] = []
                
                config_manager.reset_cohort(cohort)            
                
                cohort_final_model_dir = os.path.join(
                    config_manager.paths['final_model_dir'],
                    model_type,
                    model_cohort,
                    'final_model'
                )

                # find the experiment dir in the cohort_final_model_dir
                experiment_dirs = [d for d in os.listdir(cohort_final_model_dir) if 
                                os.path.isdir(os.path.join(cohort_final_model_dir, d))]
                if len(experiment_dirs) == 0:
                    raise ValueError(f"No experiment directories found in {cohort_final_model_dir}")
                elif len(experiment_dirs) > 1:
                    raise ValueError(f"Multiple experiment directories found in {cohort_final_model_dir}, expected only one.")
                
                experiment_dir = os.path.join(cohort_final_model_dir, experiment_dirs[0])

                filename = 'test_set_predictions'
                filepath = os.path.join(experiment_dir, filename + '.h5')
                if os.path.exists(filepath):
                    # load existing predictions
                    print(f"Loading existing predictions from {filepath}")
                    dynamic_deephit_dataset = ShardProcessor.load_dynamic_deephit_dataset_from_h5(filepath)
                else:
                    # create DataLoader for test dataset
                    # Create test_processor; note it will throw an error if the normalized test shard is missing.
                    # All possible windows will be computed
                    _, _, test_processor = create_processor_from_shards(config_manager, normalize=True)

                    # instantiate the model
                    # load result.json from config_manager.experiment_dir to get hyperparameters
                    with open(
                        os.path.join(experiment_dir, 'results.json'), 'r'
                    ) as f:
                        results_dict = json.load(f)
                    hyperparams = results_dict['best_hyperparameters']    
                    time_bins = results_dict['time_bins']
                    n_time_bins = len(time_bins) - 1

                    dynamic_deephit_model = instantiate_model(
                        config_manager, 
                        hyperparams, 
                        n_time_bins, 
                        num_risks=2, 
                        model_type=model_type,
                        test_processor=test_processor,
                        experiment_dir=experiment_dir
                        )

                    # run inference on test set - this function also saves the predictions to file
                    dynamic_deephit_dataset = inference_loop(
                        dynamic_deephit_model, 
                        test_processor, 
                        config_manager,
                        model_type=model_type,
                        experiment_dir=experiment_dir,
                        device='cuda'
                        )
                if i == 0:
                    stay_ids = dynamic_deephit_dataset['stay_ids']
                else:
                    # filter event to only stays in the current cohort
                    dynamic_deephit_dataset = filter_cohort(dynamic_deephit_dataset, stay_ids)    
                
                # calculate c-index 
                logits = dynamic_deephit_dataset['predictions']
                time_to_event = dynamic_deephit_dataset['time_to_event']
                event = dynamic_deephit_dataset['event']

                # event_label needs to be expanded so the each trajectory has a label at every timestep
                event_label = np.zeros_like(time_to_event, dtype=np.int32)
                for i in range(len(event)):
                    start_idx = dynamic_deephit_dataset['index_ptr'][i]
                    end_idx = dynamic_deephit_dataset['index_ptr'][i+1]
                    event_label[start_idx:end_idx] = event[i]
                
                if model_type == 'GRU':
                    # switch the last 2 dimesnsions of logits to match expected shape (N, D, K)
                    logits = np.transpose(logits, (0, 2, 1))

                # convert logits, time_to_event, event_label to torch tensors
                logits = torch.from_numpy(logits)
                time_to_event = torch.from_numpy(time_to_event)
                event_label = torch.from_numpy(event_label) 

                logits_dict[model_cohort] = logits
                time_to_event_dict[model_cohort] = time_to_event
                event_label_dict[model_cohort] = event_label
            
            # perform paired bootstrap
            n_bootstrap_samples = 100
            for i in range(n_bootstrap_samples):
                for i, model_cohort in enumerate(model_types):
                    logits = logits_dict[model_cohort]
                    time_to_event = time_to_event_dict[model_cohort]
                    event_label = event_label_dict[model_cohort]

                    if i == 0:
                        # generate random sample indices with replacement
                        n_samples = logits.shape[0]
                        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)

                        # downsampled indices for c-index calculation - we use 10,000 samples for speed
                        if n_samples > 10000:
                            downsampled_indices = np.random.choice(n_samples, size=10000, replace=False)
                        else:
                            downsampled_indices = np.arange(n_samples)
                    
                    c_index_results = calculate_c_index(
                        logits[downsampled_indices], 
                        time_to_event[downsampled_indices], 
                        event_label[downsampled_indices], 
                        random_sample=True
                    )
                    
                    # brier score
                    brier_score_results = calculate_brier_score(
                        logits[sample_indices],
                        time_to_event[sample_indices],
                        event_label[sample_indices]
                    )

                    # calculate roc and prc
                    roc_prc_results = calculate_roc_and_prc(
                        logits[sample_indices],
                        time_to_event[sample_indices],
                        event_label[sample_indices]
                    )

                    # calculate ece
                    ece_results = calculate_all_bin_ece(
                        logits[sample_indices].numpy(),
                        time_to_event[sample_indices].numpy(),
                        event_label[sample_indices].numpy()
                    )

                    c_index_values[model_type][cohort][model_cohort].append(c_index_results["C_weighted_multi"])
                    if i % 10 == 0:
                        print(f"Model: {model_type}, Cohort: {cohort}, Bootstrap Sample: {i}, C-index: {c_index_results['C_weighted_multi']:.4f}")



            # calculate statistical difference between between c_index_values[model_type][cohort]['all']
            # and c_index_values[model_type][cohort][cohort] using a paired t-test
            all_values = c_index_values[model_type][cohort]['all']  
            cohort_values = c_index_values[model_type][cohort][cohort]
            # run unpaired t-test
            t_stat, p_value = stats.ttest_ind(all_values, cohort_values)
            statistics[model_type][cohort]['t_stat'] = t_stat
            statistics[model_type][cohort]['p_value'] = p_value            
    
    # save the c-index values to a json file
    c_index_filepath = os.path.join(config_manager.paths['final_model_dir'], 'c_index_values.json')
    with open(c_index_filepath, 'w') as f:
        json.dump(c_index_values, f, indent=4)
    print(f"C-index values saved to {c_index_filepath}")
    
    # plot c-index values
    plot_c_index(c_index_values, statistics, plot_dir = config_manager.paths['final_model_dir'])

    # save the statistics to a json file
    statistics_filepath = os.path.join(config_manager.paths['final_model_dir'], 'c_index_statistics.json')
    with open(statistics_filepath, 'w') as f:
        json.dump(statistics, f, indent=4)
    print(f"C-index statistics saved to {statistics_filepath}")


def main():
    predict()


if __name__ == "__main__":
    main()