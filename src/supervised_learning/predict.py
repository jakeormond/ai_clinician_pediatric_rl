"""
For generating predictions using a trained Dynamic-DeepHit model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.

"""

# predict.py
import argparse
import json
from sklearn import metrics
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
    if model_type == 'GRU':
        # switch the last 2 dimesnsions of logits to match expected shape (N, D, K)
        all_predictions = np.transpose(all_predictions, (0, 2, 1))

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


def plot_c_index(c_index_values, plot_dir):
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
            

def get_model_dir(config_manager, model_type, data_cohort, model_cohort):
    config_manager.reset_cohort(data_cohort)

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
    
    return experiment_dir, cohort_final_model_dir


def get_predictions(config_manager, model_type, experiment_dir, stay_ids=None):
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
        
    if stay_ids is not None:
        # filter event to only stays in the current cohort
        dynamic_deephit_dataset = filter_cohort(dynamic_deephit_dataset, stay_ids)   

    return dynamic_deephit_dataset 


def paired_bootstrap_evaluation(
        dynamic_deephit_datasets,
        model_type,
        cohort,
        n_bootstrap_samples = 100,
):
    model_groups = dynamic_deephit_datasets.keys()
    # create storage for logits, time_to_event, event_label
    logits_dict = {}

    for i, model_group in enumerate(model_groups):
        dynamic_deephit_dataset = dynamic_deephit_datasets[model_group]

        logits = dynamic_deephit_dataset['predictions']
        logits = torch.from_numpy(logits)
        logits_dict[model_group] = logits
        
        if i == 0:
            time_to_event = dynamic_deephit_dataset['time_to_event']
            event = dynamic_deephit_dataset['event']

            # event_label needs to be expanded so the each trajectory has a label at every timestep
            event_label = np.zeros_like(time_to_event, dtype=np.int32)
            for i in range(len(event)):
                start_idx = dynamic_deephit_dataset['index_ptr'][i]
                end_idx = dynamic_deephit_dataset['index_ptr'][i+1]
                event_label[start_idx:end_idx] = event[i]
                
            time_to_event = torch.from_numpy(time_to_event)
            event_label = torch.from_numpy(event_label)

    c_index_bootstraps = {model_group: [] for model_group in model_groups}
    brier_score_bootstraps = {model_group: [] for model_group in model_groups}
    roc_prc_bootstraps = {model_group: [] for model_group in model_groups}
    ece_bootstraps = {model_group: [] for model_group in model_groups}   
    
    for i in range(n_bootstrap_samples):
        for j, model_group in enumerate(model_groups):
            logits = logits_dict[model_group]
            
            if j == 0:
                n_samples = logits.shape[0]
                # generate random sample indices with replacement                
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

            # convert tensors to floats or arrays for JSON serialization
            c_index_results = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in c_index_results.items()}
            brier_score_results = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in brier_score_results.items()}
            roc_prc_results = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in roc_prc_results.items()}
            ece_results = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in ece_results.items()}
            
            c_index_bootstraps[model_group].append(c_index_results)
            brier_score_bootstraps[model_group].append(brier_score_results)
            roc_prc_bootstraps[model_group].append(roc_prc_results)
            ece_bootstraps[model_group].append(ece_results)
            
            if i % 10 == 0:
                print(f"Model: {model_group}, Cohort: {cohort}, Bootstrap Sample: {i}, C-index: {c_index_results['C_weighted_multi']:.4f}")

    for model_group in model_groups:
        c_index_bootstraps[model_group] = {k: [d[k] for d in 
            c_index_bootstraps[model_group]] for k in 
                c_index_bootstraps[model_group][0]}
        
        brier_score_bootstraps[model_group] = {k: [d[k] for d in 
            brier_score_bootstraps[model_group]] for k in 
                brier_score_bootstraps[model_group][0]}
        
        roc_prc_bootstraps[model_group] = {k: [d[k] for d in 
            roc_prc_bootstraps[model_group]] for k in 
                roc_prc_bootstraps[model_group][0]}    

        ece_bootstraps[model_group] = {k: [d[k] for d in 
            ece_bootstraps[model_group]] for k in 
                ece_bootstraps[model_group][0]}

    return c_index_bootstraps, brier_score_bootstraps, roc_prc_bootstraps, ece_bootstraps


def get_bootstrap_storage(outer_groups, inner_groups=None):
    if inner_groups is None:
        c_index_bootstraps = {
            outer: {}
            for outer in outer_groups
            }
        
        brier_score_bootstraps = {
            outer: {}
            for outer in outer_groups
            }
        
        roc_prc_bootstraps = {
            outer: {}
            for outer in outer_groups
            }
        
        ece_bootstraps = {
            outer: {}
            for outer in outer_groups
            }          
    else:
        c_index_bootstraps = {
            outer: {
                inner: {}
                for inner in inner_groups
            }
            for outer in outer_groups
            }
        
        brier_score_bootstraps = {
            outer: {    
                inner: {}
                for inner in inner_groups
            }
            for outer in outer_groups
            }
        
        roc_prc_bootstraps = {
            outer: {
                inner: {}
                for inner in inner_groups
            }
            for outer in outer_groups
            }
        
        ece_bootstraps = {
            outer: {
                inner: {}
                for inner in inner_groups
            }
            for outer in outer_groups
            }
        
    return c_index_bootstraps, brier_score_bootstraps, roc_prc_bootstraps, ece_bootstraps


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def predict_all(
        config_manager, 
        n_bootstrap_samples=100,
        models=['GRU', 'transformer'],
        cohorts=['sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad'],
        outer_group='models',
        inner_group='cohorts',
        base_filename='all_cohorts'
    ): # for dynamic deephit models only
   
    # generate bootstrap storage
    # c_index_bootstraps, brier_score_bootstraps, roc_prc_bootstraps, ece_bootstraps \
    #     = get_bootstrap_storage(models, cohorts, outer_group, inner_group)

    if outer_group == 'models':
        outer_groups = models
        inner_groups = cohorts
        c_index_bootstraps, brier_score_bootstraps, roc_prc_bootstraps, ece_bootstraps = \
            get_bootstrap_storage(outer_groups, inner_groups=cohorts)

    else:
        outer_groups = cohorts
        inner_groups = [models]
        c_index_bootstraps, brier_score_bootstraps, roc_prc_bootstraps, ece_bootstraps = \
            get_bootstrap_storage(outer_groups)    

    for outer in outer_groups:
        if outer_group == 'models':
            model_types = [outer, outer]
                        
        else:
            model_types = models            
            # if outer group is cohort, there is only one cohort, we are comparing gru vs transformer
            model_cohorts = [outer, outer]
            model_cohorts_reversed = model_cohorts
            cohort = outer
        
        for inner in inner_groups:
            if outer_group == 'models':
                model_cohorts = ['all', inner] 
                cohort = inner    
                inner = model_cohorts[::-1]
                model_cohorts_reversed = model_cohorts[::-1]

            dynamic_deephit_datasets = {}

            for i in range(2):               
                
                experiment_dir, cohort_final_model_dir = get_model_dir(
                    config_manager, model_types[i], cohort, model_cohorts_reversed[i]
                )
                
                if i == 0:
                    dynamic_deephit_datasets[inner[i]] = get_predictions(
                        config_manager, 
                        model_types[i], 
                        experiment_dir
                    )
                    stay_ids = dynamic_deephit_datasets[inner[i]]['stay_ids']
                else:
                    dynamic_deephit_datasets[inner[i]] = get_predictions(
                        config_manager, 
                        model_types[i], 
                        experiment_dir,
                        stay_ids=stay_ids
                    )
                                
            # bootstrapped evaluation
            c_index, brier_score, roc_prc, ece = paired_bootstrap_evaluation(
                dynamic_deephit_datasets,
                model_types,
                cohort,
                n_bootstrap_samples = n_bootstrap_samples,
            )
            
            if outer_group == 'cohorts':
                for group in inner:
                    c_index_bootstraps[outer][group] = c_index[group]
                    brier_score_bootstraps[outer][group] = brier_score[group]
                    roc_prc_bootstraps[outer][group] = roc_prc[group]
                    ece_bootstraps[outer][group] = ece[group]     

            else:
                for group in inner:
                    c_index_bootstraps[outer][cohort][group] = c_index[group]
                    brier_score_bootstraps[outer][cohort][group] = brier_score[group]
                    roc_prc_bootstraps[outer][cohort][group] = roc_prc[group]
                    ece_bootstraps[outer][cohort][group] = ece[group]       

    # save the c-index values to a json file
    c_index_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_c_index_values.json')
    with open(c_index_filepath, 'w') as f:
        json.dump(_to_jsonable(c_index_bootstraps), f, indent=4)
    print(f"C-index values saved to {c_index_filepath}")

    brier_score_bootstraps_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_brier_score_values.json')
    with open(brier_score_bootstraps_filepath, 'w') as f:
        json.dump(_to_jsonable(brier_score_bootstraps), f, indent=4)
    print(f"Brier score values saved to {brier_score_bootstraps_filepath}")

    roc_prc_bootstraps_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_roc_prc_values.json')
    with open(roc_prc_bootstraps_filepath, 'w') as f:
        json.dump(_to_jsonable(roc_prc_bootstraps), f, indent=4)
    print(f"ROC and PRC values saved to {roc_prc_bootstraps_filepath}")

    ece_bootstraps_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_ece_values.json')
    with open(ece_bootstraps_filepath, 'w') as f:
        json.dump(_to_jsonable(ece_bootstraps), f, indent=4)
    print(f"ECE values saved to {ece_bootstraps_filepath}")


def get_formatted_cell(boot_samples):
    """
    Takes a list of bootstrap scores and returns a 'Mean (CI_Low - CI_High)' string.
    """
    mean = np.mean(boot_samples)
    low = np.percentile(boot_samples, 2.5)
    high = np.percentile(boot_samples, 97.5)
    
    # Using 3 decimal places is standard for clinical journals
    return f"{mean:.3f} ({low:.3f}–{high:.3f})"



def create_summary_table(c_index_values, brier_score_values, roc_prc_values, ece_values, inner_group, config_manager):
    outer_keys = c_index_values.keys()
    inner_keys = c_index_values[list(outer_keys)[0]].keys()
    if len(outer_keys) == 1:
        # only one outer key, so inner keys are cohorts
        rows = []
        for inner_key in inner_keys:
            rows.append({
                inner_group: inner_key,
                "Weighted C-Index": get_formatted_cell(c_index_values[list(outer_keys)[0]][inner_key]['C_weighted_multi']),
                
                "C-Index (Surv)": get_formatted_cell(np.array(c_index_values[list(outer_keys)[0]][inner_key]['C_integrated_k'])[:,0]),
                "C-Index (Death)": get_formatted_cell(np.array(c_index_values[list(outer_keys)[0]][inner_key]['C_integrated_k'])[:,1]),
                
                "Brier (Multi)": get_formatted_cell(brier_score_values[list(outer_keys)[0]][inner_key]['ibs_multi']),
                "Brier (Surv)": get_formatted_cell(np.array(brier_score_values[list(outer_keys)[0]][inner_key]['ibs_k'])[:,0]),
                "Brier (Death)": get_formatted_cell(np.array(brier_score_values[list(outer_keys)[0]][inner_key]['ibs_k'])[:,1]),

                "Weighted ECE": get_formatted_cell(ece_values[list(outer_keys)[0]][inner_key]['summary_ece']),
                "Integrated ECE (Surv)": get_formatted_cell(np.array(ece_values[list(outer_keys)[0]][inner_key]['event_eces'])[:,0]),
                "Integrated ECE (Death)": get_formatted_cell(np.array(ece_values[list(outer_keys)[0]][inner_key]['event_eces'])[:,1])
            })
            
        df = pd.DataFrame(rows)    





    pass


def plot_evaluation(
        config_manager,
        cohorts=['sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad'],
        models=['GRU', 'transformer'],
        outer_group='models',
        inner_group='cohorts',
        base_filename='all_stays_model_comparison'
    ):

    # load c-index values from json file
    c_index_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_c_index_values.json')
    with open(c_index_filepath, 'r') as f:
        c_index_values = json.load(f)
    print(f"C-index values loaded from {c_index_filepath}")

    brier_score_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_brier_score_values.json')
    with open(brier_score_filepath, 'r') as f:
        brier_score_values = json.load(f)
    print(f"Brier score values loaded from {brier_score_filepath}")

    roc_prc_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_roc_prc_values.json')
    with open(roc_prc_filepath, 'r') as f:
        roc_prc_values = json.load(f)
    print(f"ROC and PRC values loaded from {roc_prc_filepath}")

    ece_filepath = os.path.join(config_manager.paths['final_model_dir'], base_filename + '_ece_values.json')
    with open(ece_filepath, 'r') as f:
        ece_values = json.load(f)
    print(f"ECE values loaded from {ece_filepath}")
  
    # create table of scores
    create_summary_table(c_index_values, brier_score_values, roc_prc_values, ece_values, inner_group, config_manager)

    
    # compute p-values for each cohort between 'all' and cohort-specific models
    # plot c-index comparison
    plot_dir = config_manager.paths['final_model_dir']
    plot_c_index(c_index_values, plot_dir)


def main():
    predict()


if __name__ == "__main__":
    main()