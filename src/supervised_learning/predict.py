"""
For generating predictions using a trained Dynamic-DeepHit model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.

"""

# predict.py
import argparse
import json
from sklearn import metrics
from ..supervised_learning.plotting import plot_calibration_curve
import torch
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

from torch.utils.data import DataLoader
import torch.nn.functional as F

# from scipy.stats import 

from ..data.data_processor_for_shards import ShardProcessor
from ..rl.replay_buffer import DynamicDeepHit_Lazy, worker_init_fn
from ..supervised_learning.models.dynamic_deephit import DynamicDeepHitMulti
from ..supervised_learning.models.dynamic_deephit_transformer import TransformerDeepHitMulti
from ..supervised_learning.training import create_processor_from_shards
from ..supervised_learning.dynamic_deephit_evaluation import calculate_probabilities, calculate_c_index, calculate_brier_score, calculate_roc_and_prc, calculate_all_bin_ece


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
    dynamic_deephit_dataset['time_to_event_raw'] = dynamic_deephit_dataset['time_to_event_raw'][traj_mask]
    dynamic_deephit_dataset['time_to_event'] = dynamic_deephit_dataset['time_to_event'][traj_mask]
    if 'predictions' in dynamic_deephit_dataset:
        dynamic_deephit_dataset['predictions'] = dynamic_deephit_dataset['predictions'][traj_mask]

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
    
    feature_dim = test_processor.dynamic_deephit_dataset['feature_dim'] 
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
            feature_dim=feature_dim + 1,   # input feature dimension - add 1 for the local time feature we add in the replay buffer
            d_model=hyperparams['d_model'],               # e.g., 128 or 256
            nhead=hyperparams['nhead'],                   # e.g., 4 or 8
            num_encoder_layers=hyperparams['num_encoder_layers'],   # e.g., 3 or 6
            num_time_bins=n_time_bins,
            # num_risks=num_risks,
            num_targets=2,
            dropout=hyperparams['dropout'],
            ff_factor=hyperparams['ff_factor'],
            max_seq_len=config_manager.training_config['window_length'],
            predict_covariates=test_processor.dynamic_deephit_dataset['physiological_feature_ind'],       
        ).to(config_manager.training_config['device'])
    
    # load trained model weights from final_model.pt
    model_path = os.path.join(experiment_dir, 'final_model.pt')
    device = config_manager.training_config['device']
    
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    dynamic_deephit_model.load_state_dict(state_dict)
    
    
    # dynamic_deephit_model.load_state_dict(torch.load(
    #     model_path, 
    #     map_location=device,
    #     weights_only=True
    # ))
    dynamic_deephit_model.eval()   
    return dynamic_deephit_model

def inference_loop(
        model, 
        processor, 
        config_manager, 
        hyperparams=None,
        filename=None, 
        model_type=None, 
        experiment_dir=None,
        device='cuda'
        ):
    
    stay_ids = processor.dynamic_deephit_dataset['stay_ids']
    index_ptr = processor.dynamic_deephit_dataset['index_ptr']
    window_length = config_manager.training_config['window_length']

    response_zero_ind = processor.dynamic_deephit_dataset['response_zero_ind']
    carry_forward_ind = processor.dynamic_deephit_dataset['carry_forward_ind']

    actions = hyperparams['actions']
    
    if actions is not None:
        action_ind = []
        for action in actions:
            ind = processor.dynamic_deephit_dataset['feature_names'].index(action)
            action_ind.append(ind)

            mask_name = action + "_mask"
            if mask_name in processor.dynamic_deephit_dataset['feature_names']:
                ind2 = processor.dynamic_deephit_dataset['feature_names'].index(action + "_mask")
                action_ind.append(ind2)

            delta_name = action + "_delta"
            if delta_name in processor.dynamic_deephit_dataset['feature_names']:
                ind3 = processor.dynamic_deephit_dataset['feature_names'].index(action + "_delta")
                action_ind.append(ind3)

        response_zero_ind = np.sort(list(set(response_zero_ind) & set(action_ind)))
        carry_forward_ind = np.sort(list(set(carry_forward_ind) & set(action_ind)))
    
    # create storage for predictions
    all_predictions = []   
    all_outputs = [] 
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

            # zero out or carry forward response features at the final timestep of each window, as appropriate for the RL setting
            if len(response_zero_ind) > 0:
                windows[:, -1, response_zero_ind] = 0.0
            if len(carry_forward_ind) > 0:
                windows[:, -1, carry_forward_ind] = windows[:, -2, carry_forward_ind]
            
            # initialize loss mask (not used during inference, but required by model forward), zeros of shape (batch_size, window_length)
            # loss_mask = torch.ones((windows.shape[0], window_length), dtype=torch.float32, device=device)
            loss_mask = torch.zeros((windows.shape[0], window_length), dtype=torch.bool, device=device)

            if model_type == 'dynamic_deephit_transformer':
                # 3a. Create a (T, window_length) grid of the local sequence [0, 1, ... 71]
                local_time = torch.arange(window_length, dtype=torch.float32, device=device).repeat(T, 1)

                # 3b. Adjust for early windows (where T < window_length)
                # We want the padding at the start to hold the '0' value 
                # until the actual patient data begins.
                n_padded_windows = min(window_length - 1, T)
                for t in range(n_padded_windows):
                    # Calculate how much of this specific window is just padding
                    # e.g., if t=0 (Hour 1), pad_len is 71.
                    pad_len = window_length - (t + 1)
                    
                    # Set the padded portion to 0.0 (or the start of the stay)
                    local_time[t, :pad_len] = local_time[t, pad_len]

                # 3c. Normalize
                local_time_norm = local_time / (window_length - 1)

                # 3d. Concatenate
                windows = torch.cat([windows, local_time_norm.unsqueeze(-1)], dim=-1)
                       
            n_padded_windows = min([window_length - 1, T])
            for t in range(n_padded_windows):
                loss_mask[t, :(window_length - (t + 1))] = 0

            # 4. Batch Inference
            batch_size = 64
            all_preds = []
            all_outs = []
            
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
                    risk_logits, _, output = model(windows_chunk)
                    risk_logits = [logits[:, -1, :].cpu() for logits in risk_logits]  # unpack tuple
                    risk_logits = torch.stack(risk_logits, dim=2)  # (batch_size, num_risks, num_time_bins)

                    # check if risk_logits has any nulls
                    if torch.isnan(risk_logits).any():
                        raise ValueError(f"NaN values found in risk_logits for stay_id {stay_id} at chunk {chunk_start}-{chunk_end}")

                    # check if output has any nulls
                    if torch.isnan(output).any():
                        raise ValueError(f"NaN values found in model output for stay_id {stay_id} at chunk {chunk_start}-{chunk_end}")

                # reshape to (T, K, D)
                final_step_logits = risk_logits.permute(0, 2, 1)

                # add to list of all predictions    
                all_preds.append(final_step_logits)
                all_outs.append(output.cpu())
            
            # 5. concatenate all predictions for this patient
            patient_preds = torch.cat(all_preds, dim=0)  
            all_predictions.append(patient_preds)
            all_outputs.append(torch.cat(all_outs, dim=0))           

    # concatenate all patient predictions into a numpy array
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
    if model_type == 'GRU':
        # switch the last 2 dimesnsions of logits to match expected shape (N, D, K)
        all_predictions = np.transpose(all_predictions, (0, 2, 1))

    # calculate entropy and health score for each prediction and add to all_predictions
    entropy = calculate_prediction_entropy(torch.from_numpy(all_predictions))  # (N, T)
    
    health_score = calculate_comprehensive_health_score(torch.from_numpy(all_predictions))
    
    # add predictions to processor
    processor.dynamic_deephit_dataset['predictions'] = all_predictions
    processor.dynamic_deephit_dataset['entropy'] = entropy.cpu().numpy()
    processor.dynamic_deephit_dataset['health_score'] = health_score.cpu().numpy()
    processor.dynamic_deephit_dataset['model_outputs'] = all_outputs

    # we can exclude trajectories, event, time_to_event_raw, time_to_event from the dataset to save space, since these are already stored in the original dynamic_deephit_dataset and we have the stay_ids to link them
    keys_to_exclude = ['trajectories', 'time_to_event_raw', 'time_to_event']

    # save all values in processor.dynamic_deephit_dataset to an h5 file
    if filename is not None:
        file_path = os.path.join(experiment_dir, filename + '.h5')
    else:
        file_path = os.path.join(experiment_dir, 'predictions.h5')
    
    processor.dynamic_deephit_dataset_to_h5(file_path, keys_to_exclude=keys_to_exclude)
    dynamic_deephit_dataset = processor.dynamic_deephit_dataset

    return dynamic_deephit_dataset


def calculate_prediction_entropy(risk_logits):
    """
    Args:
        risk_logits: List of 2 tensors, each [batch, seq_len, 8]
    Returns:
        entropy: Tensor of shape [batch, seq_len]
    """
    # 1. Stack the risks: [batch, seq_len, 2, 8]
    # 2. Flatten to a single distribution: [batch, seq_len, 16]
    batch_size = risk_logits.shape[0]
    
    # 3. Convert to probabilities
    probs = calculate_probabilities(risk_logits.cpu().numpy())  # (batch, seq_len, 16)
    
    # reshape
    reshaped_probs = probs.reshape(batch_size, -1)

    # 4. Calculate Shannon Entropy: -sum(p * log(p))
    # We add a tiny epsilon to avoid log(0)
    entropy = -torch.sum(reshaped_probs * torch.log(reshaped_probs + 1e-9), dim=-1)
    
    return entropy


def calculate_comprehensive_health_score(risk_logits):
    """
    risk_logits...
    """
    # 1. Combine into a single distribution [Batch, Seq, 16]
    # Dim 2 is [Death_0...Death_7, Disc_0...Disc_7]
    batch_size, k, t = risk_logits.shape
    probs = calculate_probabilities(risk_logits)
    probs_reshaped = probs.reshape(batch_size, -1)  # (batch, seq_len, 16)

    # 2. Create the Weight Vector
    # We want a linear progression for each event type
    # Death weights: -1.0 (soon) to -0.2 (distant)
    death_weights = torch.linspace(-1.0, -0.2, steps=t)
    
    # Discharge weights: 0.4 (distant) to 1.0 (soon)
    # Note: Distant discharge (Bin 7) is index 8, Soon discharge (Bin 0) is index 15
    # after the cat() and flipping the order for Discharge
    disc_weights = torch.linspace(0.2, 1.0, steps=t)
    disc_weights = torch.flip(disc_weights, dims=[0]) # So Bin 0 is 1.0

    weights = torch.cat([disc_weights, death_weights]).to(probs.device)

    # 3. Calculate Weighted Sum (Expected Health Value)
    # Result is between -1.0 and 1.0
    raw_score = torch.sum(probs_reshaped * weights, dim=-1)

    # 4. Rescale to 0-100 for better clinical visualization
    # -1.0 -> 0 (Death Soon), 1.0 -> 100 (Discharge Soon)
    health_score = (raw_score + 1.0) * 50
    
    return health_score


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
        dynamic_deephit_dataset = inference_loop(dynamic_deephit_model, test_processor, config_manager, device='cuda')


def plot_c_index_boxplots(c_index_values, plot_dir):
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
        'disease_cohort_results',
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


def get_all_predictions(config_manager, model_type, experiment_dir):
    """
    generates predictions from that split's model for train, val and test sets. 
    These are used for training the RL model.
    """
    splits = ['train', 'val', 'test']
    predictions = {}
    for split in splits:
        print(f"Generating predictions for {split} set...")

        predictions[split] = get_predictions(config_manager, model_type, experiment_dir, set=split)

    return predictions

def get_predictions(config_manager, model_type, experiment_dir, set='test', stay_ids=None):
    filename = f'{set}_predictions'
    filepath = os.path.join(experiment_dir, filename + '.h5')
    if os.path.exists(filepath):
        # load existing predictions
        print(f"Loading existing predictions from {filepath}")
        dynamic_deephit_predictions = ShardProcessor.load_dynamic_deephit_predictions_from_h5(filepath, keys_to_exclude=['compressed_latents'])
    else:
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        processor = create_processor_from_shards(config_manager, normalize=True, set_to_return = set)

        # list folders inside experiment_dir that begin local or unknown
        # model_folders = [f for f in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, f)) and (f.startswith('local') or f.startswith('unknown'))]
        # if len(model_folders) != 1:
        #     raise ValueError(f"Expected exactly one model folder in {experiment_dir}, but found {len(model_folders)}: {model_folders}")
        # model_folder = model_folders[0]
        # experiment_dir = os.path.join(experiment_dir, model_folder)
        
        # load result.json from config_manager.experiment_dir to get hyperparameters
        with open(
            os.path.join(experiment_dir, 'results.json'), 'r'
        ) as f:
            results_dict = json.load(f)
        # hyperparams = results_dict['best_hyperparameters']    
        hyperparams = results_dict['hyperparameters']    
        # time_bins = results_dict['time_bins']
        time_bins = config_manager.time_bins
        n_time_bins = len(time_bins) - 1

        dynamic_deephit_model = instantiate_model(
            config_manager, 
            hyperparams, 
            n_time_bins, 
            num_risks=2, 
            model_type=model_type,
            test_processor=processor,
            experiment_dir=experiment_dir
            )

        # run inference on test set - this function also saves the predictions to file
        filename = f'{set}_predictions'
        dynamic_deephit_predictions = inference_loop(
            dynamic_deephit_model, 
            processor, 
            config_manager,
            hyperparams=hyperparams,
            filename=filename,
            model_type=model_type,
            experiment_dir=experiment_dir,
            device='cuda'
            )
        
    if stay_ids is not None:
        # filter event to only stays in the current cohort
        dynamic_deephit_predictions = filter_cohort(dynamic_deephit_predictions, stay_ids)   

    return dynamic_deephit_predictions 

def bootstrap_evaluation(
        dynamic_deephit_dataset,
        n_bootstrap_samples = 100,
        config_manager=None 
):
    logits = dynamic_deephit_dataset['predictions']
    logits = torch.from_numpy(logits)

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

    for i in range(n_bootstrap_samples):
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

    
    c_index_filepath = os.path.join(config_manager.experiment_dir, 'c_index_bootstrap_results.json')
    with open(c_index_filepath, 'w') as f:
        json.dump(_to_jsonable(c_index_results), f, indent=4)
    print(f"C-index bootstrap results saved to {c_index_filepath}")

    brier_score_filepath = os.path.join(config_manager.experiment_dir, 'brier_score_bootstrap_results.json')
    with open(brier_score_filepath, 'w') as f:
        json.dump(_to_jsonable(brier_score_results), f, indent=4)
    print(f"Brier score bootstrap results saved to {brier_score_filepath}")

    roc_prc_filepath = os.path.join(config_manager.experiment_dir, 'roc_prc_bootstrap_results.json')
    with open(roc_prc_filepath, 'w') as f:
        json.dump(_to_jsonable(roc_prc_results), f, indent=4)
    print(f"ROC and PRC bootstrap results saved to {roc_prc_filepath}")

    ece_filepath = os.path.join(config_manager.experiment_dir, 'ece_bootstrap_results.json')
    with open(ece_filepath, 'w') as f:
        json.dump(_to_jsonable(ece_results), f, indent=4)
    print(f"ECE bootstrap results saved to {ece_filepath}")


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


def filter_by_time_since_admission(dynamic_deephit_dataset, time_since_admission):
    index_ptr = dynamic_deephit_dataset['index_ptr']
    time_raw = dynamic_deephit_dataset['time_to_event_raw']

    n_rows = len(time_raw)
    keep_mask = np.zeros(n_rows, dtype=bool)
    keep_stay = []

    n_stays = len(index_ptr) - 1
    
    for s in range(n_stays):
        start_idx = index_ptr[s]
        end_idx = index_ptr[s + 1]

        t = time_raw[start_idx:end_idx][::-1]  # reversed

        if time_since_admission == '0-24':
            mask = (t >= 0) & (t <= 24)
        elif time_since_admission == '24-72':
            mask = (t > 24) & (t <= 72)
        else:  # '72+'
            mask = t > 72

        if mask.any():
            keep_mask[start_idx:end_idx] = mask
            keep_stay.append(True)
        else:
            keep_stay.append(False)

        # ---- Apply mask ONCE ----
    for key in ['trajectories', 'time_to_event_raw', 'time_to_event', 'predictions']:
        dynamic_deephit_dataset[key] = dynamic_deephit_dataset[key][keep_mask]

    # ---- Filter stay-level arrays ----
    dynamic_deephit_dataset['stay_ids'] = dynamic_deephit_dataset['stay_ids'][keep_stay]
    dynamic_deephit_dataset['event'] = dynamic_deephit_dataset['event'][keep_stay]

    # ---- Rebuild index_ptr (much simpler) ----
    lengths = []
    for s in range(n_stays):
        if keep_stay[s]:
            start_idx = index_ptr[s]
            end_idx = index_ptr[s + 1]
            lengths.append(keep_mask[start_idx:end_idx].sum())

    new_index_ptr = np.zeros(len(lengths) + 1, dtype=int)
    new_index_ptr[1:] = np.cumsum(lengths)
    dynamic_deephit_dataset['index_ptr'] = new_index_ptr

    return dynamic_deephit_dataset
    
    

def predict_all(
        config_manager, 
        n_bootstrap_samples=100,
        models=['GRU', 'transformer'],
        cohorts=['sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad'],
        outer_group='models',
        inner_group='cohorts',
        base_filename='all_cohorts',
        time_since_admission=None
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

            if time_since_admission is not None:
                for key in dynamic_deephit_datasets.keys():
                    dynamic_deephit_datasets[key] = filter_by_time_since_admission(
                        dynamic_deephit_datasets[key],
                        time_since_admission
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
    if time_since_admission is not None:
        base_filename = f"{base_filename}_time_since_admission_{time_since_admission}"
    
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
    
    if inner_group == 'models':
        inner_keys = [list(c_index_values[list(outer_keys)[0]].keys())]
    else:
        inner_keys_temp = list(c_index_values[list(outer_keys)[0]].keys())
        inner_keys = [[key, 'all'] for key in inner_keys_temp]
    
    if len(outer_keys) == 1:
        for inner_key in inner_keys:
            rows = []
            for key in inner_key:
                if inner_group == 'models':
                    c_index_data = c_index_values[list(outer_keys)[0]][key]
                    brier_score_data = brier_score_values[list(outer_keys)[0]][key]
                    ece_data = ece_values[list(outer_keys)[0]][key]
                    roc_prc_data = roc_prc_values[list(outer_keys)[0]][key]
                else:
                    c_index_data = c_index_values[list(outer_keys)[0]][inner_key[0]][key]
                    brier_score_data = brier_score_values[list(outer_keys)[0]][inner_key[0]][key]
                    ece_data = ece_values[list(outer_keys)[0]][inner_key[0]][key]
                    roc_prc_data = roc_prc_values[list(outer_keys)[0]][inner_key[0]][key]               
                
                rows.append({
                    inner_group: key,
                    "Weighted C-Index": get_formatted_cell(c_index_data['C_weighted_multi']),
                    
                    "C-Index (Surv)": get_formatted_cell(np.array(c_index_data['C_integrated_k'])[:,0]),
                    "C-Index (Death)": get_formatted_cell(np.array(c_index_data['C_integrated_k'])[:,1]),
                    
                    "Brier (Multi)": get_formatted_cell(brier_score_data['ibs_multi']),
                    "Brier (Surv)": get_formatted_cell(np.array(brier_score_data['ibs_k'])[:,0]),
                    "Brier (Death)": get_formatted_cell(np.array(brier_score_data['ibs_k'])[:,1]),
                    
                    "Weighted ECE": get_formatted_cell(ece_data['summary_ece']),
                    "Integrated ECE (Surv)": get_formatted_cell(np.array(ece_data['event_eces'])[:,0]),
                    "Integrated ECE (Death)": get_formatted_cell(np.array(ece_data['event_eces'])[:,1]),
                    
                    "auroc": get_formatted_cell(np.array(roc_prc_data['auroc'])),
                    "auprc": get_formatted_cell(np.array(roc_prc_data['average_precision']))
                
                })
            
            df = pd.DataFrame(rows)    
            
            if inner_group == 'models':
                table_filepath = os.path.join(config_manager.paths['final_model_dir'], 'all_models_summary_table.csv')
            else:
                table_filepath = os.path.join(config_manager.paths['final_model_dir'], f'{inner_key[0]}_summary_table.csv')
            
            df.to_csv(table_filepath, index=False)
            print(f"Summary table saved to {table_filepath}")
        
        return df

    else:
        # raise NotImplementedError for now
        raise NotImplementedError("Summary table creation for multiple outer keys is not implemented yet.")


def combine_calibration_bootstraps(calibration_bootstraps):
    combined_data = {}
    all_keys = calibration_bootstraps[0].keys() # ['0_0', '0_1', ..., '1_7']

    for key in all_keys:
        # 1. Collect all accuracy/confidence arrays for this specific key
        # Using float to ensure 'null' from JSON becomes 'np.nan'
        acc_list = [np.array(boot[key]['accuracies'], dtype=float) for boot in calibration_bootstraps]
        conf_list = [np.array(boot[key]['confidences'], dtype=float) for boot in calibration_bootstraps]
        count_list = [np.array(boot[key]['counts'], dtype=float) for boot in calibration_bootstraps]

        # 2. Stack into 2D arrays: Shape (100, 10) 
        # (100 bootstraps, 10 calibration bins)
        acc_stack = np.stack(acc_list)
        conf_stack = np.stack(conf_list)
        count_stack = np.stack(count_list)

        # 3. Calculate stats across axis 0 (the bootstrap dimension)
        combined_data[key] = {
            "mean_accuracy": np.nanmean(acc_stack, axis=0),
            "lower_accuracy": np.nanpercentile(acc_stack, 2.5, axis=0),
            "upper_accuracy": np.nanpercentile(acc_stack, 97.5, axis=0),
            
            "mean_confidence": np.nanmean(conf_stack, axis=0),
            "total_counts": np.nansum(count_stack, axis=0)
        }
    
    return combined_data


def plot_calibration_panel(ax, calibration_data, risk_idx, time_bin, title="Calibration Plot"):
    # The X-axis is the predicted probability (confidence)
    keys = list(calibration_data.keys())
    
    # 1. Identity Line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')

    for key in keys:
        key_str = f"{risk_idx}_{time_bin}"

        x = calibration_data[key][f"{risk_idx}_{time_bin}"]["mean_confidence"]
        y = calibration_data[key][f"{risk_idx}_{time_bin}"]["mean_accuracy"]
        y_low = calibration_data[key][f"{risk_idx}_{time_bin}"]["lower_accuracy"]
        y_high = calibration_data[key][f"{risk_idx}_{time_bin}"]["upper_accuracy"]

        # 2. Shaded Confidence Interval
        # Note: fill_between works best if you filter out the NaNs first
        mask = ~np.isnan(x) & ~np.isnan(y)
        ax.fill_between(x[mask], y_low[mask], y_high[mask], alpha=0.2)

        # 3. Mean Curve
        ax.plot(x[mask], y[mask], 'o-', label=key)
    ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)


def plot_ece(ece_values, inner_group, config_manager):
        outer_keys = ece_values.keys()
        
        if inner_group == 'models':
            inner_keys = [list(ece_values[list(outer_keys)[0]].keys())]
        else:
            inner_keys_temp = list(ece_values[list(outer_keys)[0]].keys())
            inner_keys = [[key, 'all'] for key in inner_keys_temp]
        
        if len(outer_keys) == 1:
            for inner_key in inner_keys:
                ece_data = {}
                calibration_data = {}
                
                for key in inner_key:
                    if inner_group == 'models':
                        ece_data[key] = ece_values[list(outer_keys)[0]][key]
                    else:
                        ece_data[key] = ece_values[list(outer_keys)[0]][inner_key[0]][key]
                
                    ece_matrix_bootstraps = np.array(ece_data[key]['ece_matrix'])
                    calibration_bootstraps = ece_data[key]['curve_data']
                    # calculate mean and 95% CI for each risk
                    for risk_idx, risk_name in enumerate(['Survival', 'Death']):
                        mean_ece = np.mean(ece_matrix_bootstraps[:, risk_idx, :], axis=0)
                        ci_lower = np.percentile(ece_matrix_bootstraps[:, risk_idx, :], 2.5, axis=0)
                        ci_upper = np.percentile(ece_matrix_bootstraps[:, risk_idx, :], 97.5, axis=0)
                    
                        ece_data[key][risk_name] = {
                            'mean_ece': mean_ece,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper
                        }

                    calibration_data[key] = combine_calibration_bootstraps(calibration_bootstraps)

                # plot ece
                if inner_group == 'models':
                    plot_path = os.path.join(
                        config_manager.paths['final_model_dir'], 
                        f'ece_{list(outer_keys)[0]}.png'
                    )
                else:
                    plot_path = os.path.join(
                        config_manager.paths['final_model_dir'], 
                        f'ece_{inner_key[0]}.png'
                    )

                # 2 subplots, one for each risk
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                for risk_idx, risk_name in enumerate(['Survival', 'Death']):
                    ax = axes[risk_idx, 0]
                    for key in inner_key:
                        mean_ece = ece_data[key][risk_name]['mean_ece']
                        ci_lower = ece_data[key][risk_name]['ci_lower']
                        ci_upper = ece_data[key][risk_name]['ci_upper']

                        ax.plot( 
                            mean_ece, 
                            label=f'Cohort: {key}'
                        )
                        ax.fill_between(
                            np.arange(len(mean_ece)), 
                            ci_lower, 
                            ci_upper, 
                            alpha=0.3
                        )
                    ax.set_title(f'Expected Calibration Error - {risk_name}')
                    ax.set_xlabel('time bins')
                    ax.set_ylabel('ECE')
                    ax.legend()

                    ax = axes[risk_idx, 1]
                    plot_calibration_panel(ax, calibration_data, risk_idx, time_bin=0, title=f"{risk_name}: Immediate Horizon (0-6h)")
                    ax = axes[risk_idx, 2]
                    plot_calibration_panel(ax, calibration_data, risk_idx, time_bin=7, title=f"{risk_name}: Distal Horizon (322-1435h)")

                plt.tight_layout()
                plt.savefig(plot_path)
                print(f"ECE plot saved to {plot_path}")
                plt.close()
        
        else:
            raise NotImplementedError("ECE plotting for multiple outer keys is not implemented yet.")
                               



def plot_c_index(c_index_values, inner_group,config_manager):
    outer_keys = c_index_values.keys()
    
    if inner_group == 'models':
        inner_keys = [list(c_index_values[list(outer_keys)[0]].keys())]
    else:
        inner_keys_temp = list(c_index_values[list(outer_keys)[0]].keys())
        inner_keys = [[key, 'all'] for key in inner_keys_temp]

    if len(outer_keys) == 1:
        for inner_key in inner_keys:
            mean_c_index = {}
            ci_lower = {}
            ci_upper = {}

            for key in inner_key:
                # plot c-index over time for each cohort
                if inner_group == 'models':
                    c_index_matrix_bootstraps = np.array(c_index_values[list(outer_keys)[0]][key]['C_k_t'])
                else:
                    c_index_matrix_bootstraps = np.array(c_index_values[list(outer_keys)[0]][inner_key[0]][key]['C_k_t'])
                
                mean_c_index[key] = np.mean(c_index_matrix_bootstraps, axis=0)
                ci_lower[key] = np.percentile(c_index_matrix_bootstraps, 2.5, axis=0)
                ci_upper[key] = np.percentile(c_index_matrix_bootstraps, 97.5, axis=0)

            if inner_group == 'models':
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'c_index_over_time_{list(outer_keys)[0]}.png'
                )
            else:
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'c_index_over_time_{inner_key[0]}.png'
                )
            
            fig, axes = plt.subplots(2, 1, figsize=(8, 12))
            for risk_idx, risk_name in enumerate(['Survival', 'Death']):
                ax = axes[risk_idx]
                for key in inner_key:
                    mean_c_idx = mean_c_index[key][risk_idx, :]
                    ci_low = ci_lower[key][risk_idx, :]
                    ci_up = ci_upper[key][risk_idx, :]

                    ax.plot( 
                        mean_c_idx, 
                        label=f'Model: {key}'
                    )
                    ax.fill_between(
                        np.arange(len(mean_c_idx)), 
                        ci_low, 
                        ci_up, 
                        alpha=0.3
                    )
                ax.set_title(f'C-Index Over Time - {risk_name}')
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('C-Index')
                ax.set_ylim([0.65, .9])
                ax.grid(True, alpha=0.3)
                ax.legend()

            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"C-index plot saved to {plot_path}")
            plt.close()
            
        
    else:
        raise NotImplementedError("C-index plotting for multiple outer keys is not implemented yet.")
    

def plot_brier_score(brier_score_values, inner_group, config_manager):
    outer_keys = brier_score_values.keys()
    
    if inner_group == 'models':
        inner_keys = [list(brier_score_values[list(outer_keys)[0]].keys())]
    else:
        inner_keys_temp = list(brier_score_values[list(outer_keys)[0]].keys())
        inner_keys = [[key, 'all'] for key in inner_keys_temp]

    if len(outer_keys) == 1:
        for inner_key in inner_keys:
            mean_brier_score = {}
            ci_lower = {}
            ci_upper = {}

            for key in inner_key:
                # plot brier score over time for each cohort
                if inner_group == 'models':
                    brier_score_matrix_bootstraps = np.array(brier_score_values[list(outer_keys)[0]][key]['bs_k_t'])
                else:
                    brier_score_matrix_bootstraps = np.array(brier_score_values[list(outer_keys)[0]][inner_key[0]][key]['bs_k_t'])
                
                mean_brier_score[key] = np.mean(brier_score_matrix_bootstraps, axis=0)
                ci_lower[key] = np.percentile(brier_score_matrix_bootstraps, 2.5, axis=0)
                ci_upper[key] = np.percentile(brier_score_matrix_bootstraps, 97.5, axis=0)

            if inner_group == 'models':
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'brier_score_over_time_{list(outer_keys)[0]}.png'
                )
            else:
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'brier_score_over_time_{inner_key[0]}.png'
                )

            fig, axes = plt.subplots(2, 1, figsize=(8, 12))
            for risk_idx, risk_name in enumerate(['Survival', 'Death']):
                ax = axes[risk_idx]
                for key in inner_key:
                    mean_brier = mean_brier_score[key][risk_idx, :]
                    ci_low = ci_lower[key][risk_idx, :]
                    ci_up = ci_upper[key][risk_idx, :]

                    ax.plot( 
                        mean_brier, 
                        label=f'Model: {key}'
                    )
                    ax.fill_between(
                        np.arange(len(mean_brier)), 
                        ci_low, 
                        ci_up, 
                        alpha=0.3
                    )
                ax.set_title(f'Brier Score Over Time - {risk_name}')
                ax.set_xlabel('Time Bins')
                ax.set_ylabel('Brier Score')
                ax.set_ylim([0.0, 0.3])
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Brier score plot saved to {plot_path}")
            plt.close()
        
    else:
        raise NotImplementedError("Brier score plotting for multiple outer keys is not implemented yet.")


def plot_roc_prc(roc_prc_values, inner_group, config_manager):
    outer_keys = roc_prc_values.keys()
    
    if inner_group == 'models':
        inner_keys = [list(roc_prc_values[list(outer_keys)[0]].keys())]
    else:
        inner_keys_temp = list(roc_prc_values[list(outer_keys)[0]].keys())
        inner_keys = [[key, 'all'] for key in inner_keys_temp]

    if len(outer_keys) == 1:
        for inner_key in inner_keys:
            auroc = {}
            auprc = {}
            prevalence = {}
           
            for key in inner_key:
                # plot auroc and auprc over time for each cohort
                if inner_group == 'models':
                    auroc_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][key]['auroc'])
                    auprc_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][key]['average_precision'])
                    prevalence_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][key]['prevalence'])
                else:
                    auroc_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][inner_key[0]][key]['auroc'])
                    auprc_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][inner_key[0]][key]['average_precision'])
                    prevalence_bootstraps = np.array(roc_prc_values[list(outer_keys)[0]][inner_key[0]][key]['prevalence'])
                
                auroc[key] = auroc_bootstraps
                auprc[key] = auprc_bootstraps
                prevalence[key] = prevalence_bootstraps

                # confirm that prevalence is the same across bootstraps
            if not np.array_equal(prevalence[inner_key[0]], prevalence[inner_key[1]]):
                raise ValueError("Prevalence values differ across models, cannot plot AUPRC with prevalence line.")

            prevalence_bootstraps = prevalence[inner_key[0]]

            if inner_group == 'models':
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'roc_prc_{list(outer_keys)[0]}.png'
                )
            else:
                plot_path = os.path.join(
                    config_manager.paths['final_model_dir'], 
                    f'roc_prc_{inner_key[0]}.png'
                )

            fig, axes = plt.subplots(2, 1, figsize=(8, 12))
            
            # AUROC violins
            ax = axes[0]
            df_auroc = pd.DataFrame({
                "value": np.r_[auroc[inner_key[0]], auroc[inner_key[1]]],
                "model": [inner_key[0]] * len(auroc[inner_key[0]]) + [inner_key[1]] * len(auroc [inner_key[1]])
            })
            sns.violinplot(data=df_auroc, x="model", y="value", cut=0, inner=None, ax=ax, palette="Set2")
            sns.pointplot(data=df_auroc, x="model", y="value", estimator=np.median,
                          join=False, color="k", markers="_", errwidth=2, scale=1.5, ax=ax)
            
            # Add 95% CI as horizontal lines
            # for i, key in enumerate(inner_key):
            #     ci_low = np.percentile(auroc[key], 2.5)
            #     ci_high = np.percentile(auroc[key], 97.5)
            #     ax.plot([i - 0.2, i + 0.2], [ci_low, ci_low], 'r-', linewidth=2, alpha=0.7)
            #     ax.plot([i - 0.2, i + 0.2], [ci_high, ci_high], 'r-', linewidth=2, alpha=0.7)
            
            # get min and max for y-axis limits
            min_auroc = min(np.min(auroc[inner_key[0]]), np.min(auroc[inner_key[1]]))
            max_auroc = max(np.max(auroc[inner_key[0]]), np.max(auroc[inner_key[1]]))
            ax.set_ylim([min_auroc - 0.05, max_auroc + 0.05])

            ax.set_title("AUROC Distribution", fontsize=12, fontweight='bold')
            ax.set_ylabel("AUROC")
            ax.grid(True, alpha=0.3, axis='y')
            
            # AUPRC violins
            ax = axes[1]
            df_auprc = pd.DataFrame({
                "value": np.r_[auprc[inner_key[0]], auprc[inner_key[1]]],
                "model": [inner_key[0]] * len(auprc[inner_key[0]]) + [inner_key[1]] * len(auprc[inner_key[1]])
            })
            sns.violinplot(data=df_auprc, x="model", y="value", cut=0, inner=None, ax=ax, palette="Set2")
            sns.pointplot(data=df_auprc, x="model", y="value", estimator=np.median,
                          join=False, color="k", markers="_", errwidth=2, scale=1.5, ax=ax)
            
            # # Add 95% CI as horizontal lines
            # for i, key in enumerate(inner_key):
            #     ci_low = np.percentile(auprc[key], 2.5)
            #     ci_high = np.percentile(auprc[key], 97.5)
            #     ax.plot([i - 0.2, i + 0.2], [ci_low, ci_low], 'r-', linewidth=2, alpha=0.7)
            #     ax.plot([i - 0.2, i + 0.2], [ci_high, ci_high], 'r-', linewidth=2, alpha=0.7)
            
            # add line for prevalence
            prevalence_mean = np.mean(prevalence_bootstraps)
            prevalence_5th = np.percentile(prevalence_bootstraps, 2.5)
            prevalence_95th = np.percentile(prevalence_bootstraps, 97.5)
            ax.axhline(prevalence_mean, color='r', linestyle='--', label='Prevalence Mean')
            ax.fill_between(
                x=[-0.5, 1.5], 
                y1=prevalence_5th, 
                y2=prevalence_95th, 
                color='r', 
                alpha=0.2,
                label='Prevalence 95% CI'
            )
            ax.legend()            
            
            ax.set_title("AUPRC Distribution", fontsize=12, fontweight='bold')
            ax.set_ylabel("AUPRC")
            min_auprc = min(np.min(auprc[inner_key[0]]), np.min(auprc[inner_key[1]]))
            max_auprc = max(np.max(auprc[inner_key[0]]), np.max(auprc[inner_key[1]]))
            ax.set_ylim([prevalence_5th - 0.05, max_auprc + 0.05])
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"ROC and PRC plot saved to {plot_path}")
            plt.close()

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

    # plot calibration curves
    plot_ece(ece_values, inner_group, config_manager)

    # compute p-values for each cohort between 'all' and cohort-specific models
    # plot c-index comparison
    plot_c_index(c_index_values, inner_group, config_manager)

    # plot brier score comparison
    plot_brier_score(brier_score_values, inner_group, config_manager)

    plot_roc_prc(roc_prc_values, inner_group, config_manager)
    pass



def plot_health_score_statistics(
        config_manager,
        model='transformer',
        cohort='all',
    ):

    filename = 'test_predictions'
    filepath = os.path.join(config_manager.experiment_dir, filename + '.h5')
    # load existing predictions
    print(f"Loading existing predictions from {filepath}")
    dynamic_deephit_predictions = ShardProcessor.load_dynamic_deephit_predictions_from_h5(filepath, keys_to_exclude=['compressed_latents'])

    if config_manager.unit != 'all' and config_manager.cohort != 'all':
        dataset_name = f"test_{config_manager.cohort}_{config_manager.unit.lower()}_dynamic_deephit_dataset.h5"
    elif config_manager.unit != 'all' and config_manager.cohort == 'all':
        dataset_name = f"test_{config_manager.unit.lower()}_dynamic_deephit_dataset.h5"
    else:
        dataset_name = "test_dynamic_deephit_dataset.h5"       
    dataset_filepath = os.path.join(config_manager.get_data_dir, config_manager.data_config['shard_dir'], 
                f"split_{config_manager.split_num}", "training_set", dataset_name)    
    dynamic_deephit_dataset = ShardProcessor.load_dynamic_deephit_dataset_from_h5(dataset_filepath)
    # plot predictions for a random sample of 10 patients
    stay_ids = dynamic_deephit_dataset['stay_ids']
    events = dynamic_deephit_dataset['event']

    # get indices of patients who died and survived
    death_events = [i for i, e in enumerate(events) if e == 1]
    survival_events = [i for i, e in enumerate(events) if e == 0]

    death_start_indices = dynamic_deephit_dataset['index_ptr'][death_events]
    death_end_indices = dynamic_deephit_dataset['index_ptr'][[death_event + 1 for death_event in death_events]]

    survival_start_indices = dynamic_deephit_dataset['index_ptr'][survival_events]
    survival_end_indices = dynamic_deephit_dataset['index_ptr'][[survival_event + 1 for survival_event in survival_events]]

    health_scores_death = []    
    for start_idx, end_idx in zip(death_start_indices, death_end_indices):
        health_scores_death.append(dynamic_deephit_predictions['health_score'][start_idx:end_idx])
    
    health_scores_survival = []
    for start_idx, end_idx in zip(survival_start_indices, survival_end_indices):
        health_scores_survival.append(dynamic_deephit_predictions['health_score'][start_idx:end_idx])

    # plot distribution of health scores as overlapping kde plots
    health_scores_death_cat = np.concatenate(health_scores_death)
    health_scores_survival_cat = np.concatenate(health_scores_survival)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(health_scores_death_cat, fill=True, alpha=0.5, label='Death')
    sns.kdeplot(health_scores_survival_cat, fill=True, alpha=0.5, label='Survival')
    plt.xlabel('Health Score')
    plt.ylabel('Density')
    plt.title('Distribution of Health Scores for Death vs Survival Events')
    plt.legend()
    #plt.show()
    # save plot
    plt.savefig(os.path.join(config_manager.experiment_dir, 'health_score_distribution.png'))
    plt.close()

    # calculate non-parametric t-test (mannwhitneyu) between the two distributions
    stat, p_value = stats.mannwhitneyu(health_scores_death_cat, health_scores_survival_cat, alternative='two-sided')
    print(f"Mann-Whitney U test statistic: {stat:.3f}, p-value: {p_value:.3e}")
    
    # for each trajectory in health_scores_death and health_scores_survival, 
    # calculate the diff between each consecutive step and plot the distribution of these diffs as well
    health_score_diffs_death = []
    for trajectory in health_scores_death:
        diffs = np.diff(trajectory)
        health_score_diffs_death.append(diffs)
    
    health_score_diffs_survival = []
    for trajectory in health_scores_survival:
        diffs = np.diff(trajectory)
        health_score_diffs_survival.append(diffs)

    # plot distribution of health score diffs as overlapping kde plots
    health_score_diffs_death_cat = np.concatenate(health_score_diffs_death)
    health_score_diffs_survival_cat = np.concatenate(health_score_diffs_survival)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(health_score_diffs_death_cat, fill=True, alpha=0.5, label='Death')
    sns.kdeplot(health_score_diffs_survival_cat, fill=True, alpha=0.5, label='Survival')
    plt.xlabel('Health Score Diff')
    plt.ylabel('Density')
    plt.title('Distribution of Health Score Diffs for Death vs Survival Events')
    plt.legend()
    # plt.show()
    # save plot
    plt.savefig(os.path.join(config_manager.experiment_dir, 'health_score_diff_distribution.png'))
    plt.close()

    stat_diff, p_value_diff = stats.mannwhitneyu(health_score_diffs_death_cat, health_score_diffs_survival_cat, alternative='two-sided')
    print(f"Mann-Whitney U test for health score diffs: statistic: {stat_diff:.3f}, p-value: {p_value_diff:.3e}")

    # for each trajectory, calculate the difference between the last health score and the first health score, and plot the distribution of these diffs as well
    health_score_overall_diffs_death = []
    for trajectory in health_scores_death:
        overall_diff = trajectory[-1] - trajectory[0]
        health_score_overall_diffs_death.append(overall_diff)
    
    health_score_overall_diffs_survival = []
    for trajectory in health_scores_survival:
        overall_diff = trajectory[-1] - trajectory[0]
        health_score_overall_diffs_survival.append(overall_diff)

    # plot distribution of overall health score diffs as overlapping kde plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(health_score_overall_diffs_death, fill=True, alpha=0.5, label='Death')
    sns.kdeplot(health_score_overall_diffs_survival, fill=True, alpha=0.5, label='Survival')
    plt.xlabel('Overall Health Score Diff')
    plt.ylabel('Density')
    plt.title('Distribution of Overall Health Score Diffs for Death vs Survival Events')
    plt.legend()
    # plt.show()
    # save plot
    plt.savefig(os.path.join(config_manager.experiment_dir, 'health_score_overall_diff_distribution.png'))
    plt.close()
    
    stat_overall_diff, p_value_overall_diff = stats.mannwhitneyu(health_score_overall_diffs_death, health_score_overall_diffs_survival, alternative='two-sided')
    print(f"Mann-Whitney U test for overall health score diffs: statistic: {stat_overall_diff:.3f}, p-value: {p_value_overall_diff:.3e}")

    pass

def plot_random_predictions(
        config_manager,
        model='transformer',
        cohort='all',
    ):
    
    # experiment_dir, cohort_final_model_dir = get_model_dir(
    #     config_manager, model, cohort, cohort
    # )

    plot_dir = os.path.join(config_manager.experiment_dir, 'prediction_plots')
    os.makedirs(plot_dir, exist_ok=True)

    filename = 'test_predictions'
    filepath = os.path.join(config_manager.experiment_dir, filename + '.h5')
    # load existing predictions
    print(f"Loading existing predictions from {filepath}")
    dynamic_deephit_predictions = ShardProcessor.load_dynamic_deephit_predictions_from_h5(filepath, keys_to_exclude=['compressed_latents'])
    
    # loading the dataset to get trajectories
    if config_manager.cohort != 'all' and config_manager.unit != 'all':
        dataset_name = f"test_{config_manager.cohort}_{config_manager.unit.lower()}_dynamic_deephit_dataset.h5"
    elif config_manager.cohort == 'all' and config_manager.unit == 'PICU':
        dataset_name = f"test_{config_manager.unit.lower()}_dynamic_deephit_dataset.h5"
    else:
        dataset_name = "test_dynamic_deephit_dataset.h5"
    dataset_filepath = os.path.join(config_manager.get_data_dir, config_manager.data_config['shard_dir'], 
                f"split_{config_manager.split_num}", "training_set", dataset_name)    
    dynamic_deephit_dataset = ShardProcessor.load_dynamic_deephit_dataset_from_h5(dataset_filepath)
    
    # confirm that the predictions and dataset are aligned by index_ptr
    if not np.array_equal(dynamic_deephit_predictions['index_ptr'], dynamic_deephit_dataset['index_ptr']):
        raise ValueError("Index pointers in predictions and dataset do not match. Cannot plot predictions.")
    else:
        print("Index pointers in predictions and dataset match. Proceeding with plotting.")

    # check that trajectories and predictions have same number of rows
    if dynamic_deephit_predictions['predictions'].shape[0] != dynamic_deephit_dataset['trajectories'].shape[0]:
        raise ValueError("Number of prediction rows does not match number of trajectory rows. Cannot plot predictions.")
    else:
        print("Number of prediction rows matches number of trajectory rows. Proceeding with plotting.")
    
    
    # plot predictions for a random sample of 10 patients
    stay_ids = dynamic_deephit_dataset['stay_ids']
    events = dynamic_deephit_dataset['event']

    death_events = [i for i, e in enumerate(events) if e == 1]
    survival_events = [i for i, e in enumerate(events) if e == 0]

    # randomly sample 50 of each event type
    np.random.seed(42)
    n_samples_per_death = np.min([len(death_events), 50])
    n_samples_per_survival = np.min([len(survival_events), 50])
    sampled_death_indices = np.random.choice(death_events, size=n_samples_per_death, replace=False)
    sampled_survival_indices = np.random.choice(survival_events, size=n_samples_per_survival, replace=False)

    sampled_indices = np.concatenate([sampled_death_indices, sampled_survival_indices])

    for idx in sampled_indices:
        stay_id = stay_ids[idx]
        event = dynamic_deephit_dataset['event'][idx]

        start_idx = dynamic_deephit_dataset['index_ptr'][idx]
        end_idx = dynamic_deephit_dataset['index_ptr'][idx + 1]

        predictions = dynamic_deephit_predictions['predictions'][start_idx:end_idx]
        probabilities = calculate_probabilities(predictions)

        entropy = dynamic_deephit_predictions['entropy'][start_idx:end_idx]
        health_score = dynamic_deephit_predictions['health_score'][start_idx:end_idx]

        patient_data = {
            'stay_id': stay_id,
            'event': event,
            'time_bins': config_manager.time_bins,
            'probabilities': probabilities,
            'entropy': entropy,
            'health_score': health_score
        }   

        plot_individual_patient_predictions(
            patient_data,
            plot_dir = plot_dir
        )
        print(f"Prediction plot for patient {stay_id} saved to {plot_dir}")


def plot_individual_patient_predictions(        
        patient_data,
        plot_dir = None,
        plot_entropy=False,
    ):
    stay_id = patient_data['stay_id']
    event = patient_data['event']
    bin_edges = patient_data['time_bins']
    probs = patient_data['probabilities']
    entropy = patient_data['entropy']
    health_score = patient_data['health_score']

    event_types = ['Survival', 'Death']
    event_type = event_types[event]

    plot_title = f"Patient ID: {stay_id} | Event: {event_type}"

    # 1. Reshape for heatmap: (16 bins) x (timesteps)
    timesteps = probs.shape[0]
    heatmap_data = probs.permute(1, 2, 0).reshape(16, -1).detach().cpu().numpy()

    # 2. Create Labels for the Y-axis
    y_labels = []
    for event in event_types:
        for i in range(len(bin_edges) - 1):
            y_labels.append(f"{event} [{bin_edges[i]}-{bin_edges[i+1]}h]")

    # 3. Create 2 subplots: predictions heatmap (top) + health score / entropy (bottom)
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[40, 1],   # main plots, colorbar
        height_ratios=[2, 1],
        hspace=0.25,
        wspace=0.05
    )

    # ===== SUBPLOT 1: Predictions Heatmap =====
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_entropy = fig.add_subplot(gs[1, 0])   # same-width column as heatmap
    cax = fig.add_subplot(gs[0, 1])          # separate colorbar axis

    sns.heatmap(
        heatmap_data,
        xticklabels=False,
        yticklabels=y_labels,
        cmap="YlGnBu",
        norm=LogNorm(vmin=1e-4, vmax=heatmap_data.max()),
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": "Joint Probability $P(T, E)$"},
        ax=ax_heatmap
    )

    ax_heatmap.axhline(y=8, color='red', linewidth=2, linestyle='--')
    ax_heatmap.set_title(f"Model Predictions - {plot_title}", fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel("Event & Forecast Horizon", fontsize=11)

    # ===== SUBPLOT 2: Health Score (+ optional Entropy on secondary axis) =====
    health_score_np = health_score.cpu().numpy() if isinstance(health_score, torch.Tensor) else health_score

    x = np.arange(len(health_score_np)) + 0.5

    ax_entropy.set_xlabel('ICU Stay Duration (Hours)', fontsize=11, fontweight='bold')

    ax_health = None
    if plot_entropy:
        entropy_np = entropy.cpu().numpy() if isinstance(entropy, torch.Tensor) else entropy

        # Plot entropy on left y-axis
        color_entropy = 'steelblue'
        ax_entropy.set_ylabel('Entropy', color=color_entropy, fontsize=11, fontweight='bold')

        line1 = ax_entropy.plot(x, entropy_np, color=color_entropy, linewidth=2.5, label='Entropy', zorder=3)
        ax_entropy.fill_between(x, entropy_np, entropy_np, alpha=0.2, color=color_entropy)

        ax_entropy.tick_params(axis='y', labelcolor=color_entropy)
        ax_entropy.grid(True, alpha=0.3, zorder=0)

        # Create second y-axis for health_score
        ax_health = ax_entropy.twinx()
        color_health = 'coral'
        ax_health.set_ylabel('Health Score (0-100)', color=color_health, fontsize=11, fontweight='bold')

        line2 = ax_health.plot(x, health_score_np, color=color_health, linewidth=2.5, label='Health Score', zorder=3)
        ax_health.fill_between(x, health_score_np, health_score_np, alpha=0.2, color=color_health)

        ax_health.tick_params(axis='y', labelcolor=color_health)
        ax_health.set_ylim([0, 100])

        # Add legend combining both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_entropy.legend(lines, labels, loc='upper left', fontsize=10)
    else:
        # Single-axis health score only
        color_health = 'coral'
        ax_entropy.set_ylabel('Health Score (0-100)', color=color_health, fontsize=11, fontweight='bold')

        line2 = ax_entropy.plot(x, health_score_np, color=color_health, linewidth=2.5, label='Health Score', zorder=3)
        ax_entropy.fill_between(x, health_score_np, health_score_np, alpha=0.2, color=color_health)

        ax_entropy.tick_params(axis='y', labelcolor=color_health)
        ax_entropy.set_ylim([0, 100])
        ax_entropy.grid(True, alpha=0.3, zorder=0)
        ax_entropy.legend(loc='upper left', fontsize=10)

    # Configure X-axis for both
    num_timesteps = heatmap_data.shape[1]
    ax_entropy.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(x)}"

    ax_entropy.xaxis.set_major_formatter(major_formatter)
    ax_entropy.tick_params(axis='x', which='major', bottom=True, labelbottom=True, rotation=0)

    # Remove default 5% x padding on line plots
    ax_entropy.margins(x=0)
    if ax_health is not None:
        ax_health.margins(x=0)

    ax_entropy.set_xlim(0, len(health_score_np))

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{stay_id}_{event_type}_prediction_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()


# def plot_individual_patient_predictions(        
#         patient_data,
#         plot_dir = None,
#         plot_entropy=False,
#     ):
#     stay_id = patient_data['stay_id']
#     event = patient_data['event']
#     bin_edges = patient_data['time_bins']
#     probs = patient_data['probabilities']
#     entropy = patient_data['entropy']
#     health_score = patient_data['health_score']

#     event_types = ['Survival', 'Death']
#     event_type = event_types[event]

#     plot_title = f"Patient ID: {stay_id} | Event: {event_type}"

#     # 1. Reshape for heatmap: (16 bins) x (timesteps)
#     timesteps = probs.shape[0]
#     heatmap_data = probs.permute(1, 2, 0).reshape(16, -1).detach().cpu().numpy()

#     # 2. Create Labels for the Y-axis
#     y_labels = []
#     for event in event_types:
#         for i in range(len(bin_edges) - 1):
#             y_labels.append(f"{event} [{bin_edges[i]}-{bin_edges[i+1]}h]")

#     # 3. Create 2 subplots: predictions heatmap (top) + dual-axis entropy/health_score (bottom)
#     # fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
#     fig = plt.figure(figsize=(10, 6))
#     gs = fig.add_gridspec(
#         2, 2,
#         width_ratios=[40, 1],   # main plots, colorbar
#         height_ratios=[2, 1],
#         hspace=0.25,
#         wspace=0.05
#     )

#     # ===== SUBPLOT 1: Predictions Heatmap =====
#     # ax_heatmap = axes[0]
#     ax_heatmap = fig.add_subplot(gs[0, 0])
#     ax_entropy = fig.add_subplot(gs[1, 0])   # same-width column as heatmap
#     cax = fig.add_subplot(gs[0, 1])          # separate colorbar axis
    
#     # sns.heatmap(
#     #     heatmap_data, 
#     #     xticklabels=False,
#     #     yticklabels=y_labels,
#     #     cmap="YlGnBu", 
#     #     norm=LogNorm(vmin=1e-4, vmax=heatmap_data.max()),
#     #     cbar_kws={'label': 'Joint Probability $P(T, E)$'},
#     #     ax=ax_heatmap
#     # )
    
#     sns.heatmap(
#         heatmap_data,
#         xticklabels=False,
#         yticklabels=y_labels,
#         cmap="YlGnBu",
#         norm=LogNorm(vmin=1e-4, vmax=heatmap_data.max()),
#         cbar=True,
#         cbar_ax=cax,
#         cbar_kws={"label": "Joint Probability $P(T, E)$"},
#         ax=ax_heatmap
#     )
    
    
#     ax_heatmap.axhline(y=8, color='red', linewidth=2, linestyle='--')
#     ax_heatmap.set_title(f"Model Predictions - {plot_title}", fontsize=12, fontweight='bold')
#     ax_heatmap.set_ylabel("Event & Forecast Horizon", fontsize=11)
    
#     # ===== SUBPLOT 2: Dual-Axis Entropy & Health Score =====
#    #  ax_entropy = axes[1]
    
#     # Convert tensors to numpy if needed
#     entropy_np = entropy.cpu().numpy() if isinstance(entropy, torch.Tensor) else entropy
#     health_score_np = health_score.cpu().numpy() if isinstance(health_score, torch.Tensor) else health_score
    
#     # Plot entropy on left y-axis
#     color_entropy = 'steelblue'
#     ax_entropy.set_xlabel('ICU Stay Duration (Hours)', fontsize=11, fontweight='bold')
#     ax_entropy.set_ylabel('Entropy', color=color_entropy, fontsize=11, fontweight='bold')
    
#     # x = np.arange(len(entropy_np))
#     x = np.arange(len(entropy_np)) + 0.5
    
#     # line1 = ax_entropy.plot(entropy_np, color=color_entropy, linewidth=2.5, label='Entropy', zorder=3)
#     line1 = ax_entropy.plot(x, entropy_np, color=color_entropy, linewidth=2.5, label='Entropy', zorder=3)
    
#     # ax_entropy.fill_between(range(len(entropy_np)), entropy_np, alpha=0.2, color=color_entropy)
#     ax_entropy.fill_between(x, entropy_np, entropy_np, alpha=0.2, color=color_entropy)
        
#     ax_entropy.tick_params(axis='y', labelcolor=color_entropy)
#     ax_entropy.grid(True, alpha=0.3, zorder=0)
    
#     # Create second y-axis for health_score
#     ax_health = ax_entropy.twinx()
#     color_health = 'coral'
#     ax_health.set_ylabel('Health Score (0-100)', color=color_health, fontsize=11, fontweight='bold')
    
#     # line2 = ax_health.plot(health_score_np, color=color_health, linewidth=2.5, label='Health Score', zorder=3)
#     line2 = ax_health.plot(x, health_score_np, color=color_health, linewidth=2.5, label='Health Score', zorder=3)
    
#     # ax_health.fill_between(range(len(health_score_np)), health_score_np, alpha=0.2, color=color_health)
#     ax_health.fill_between(x, health_score_np, health_score_np, alpha=0.2, color=color_health)
        
#     ax_health.tick_params(axis='y', labelcolor=color_health)
#     ax_health.set_ylim([0, 100])
    
#     # Add legend combining both axes
#     lines = line1 + line2
#     labels = [l.get_label() for l in lines]
#     ax_entropy.legend(lines, labels, loc='upper left', fontsize=10)
    
#     # Configure X-axis for both
#     num_timesteps = heatmap_data.shape[1]
#     ax_entropy.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))
    
#     @ticker.FuncFormatter
#     def major_formatter(x, pos):
#         return f"{int(x)}"
    
#     ax_entropy.xaxis.set_major_formatter(major_formatter)
#     ax_entropy.tick_params(axis='x', which='major', bottom=True, labelbottom=True, rotation=0)
    
#     # Align x-axis between both subplots
#     # ax_heatmap.set_xlim(ax_entropy.get_xlim())

#     # Remove default 5% x padding on line plots
#     ax_entropy.margins(x=0)
#     ax_health.margins(x=0)

#     # ax_entropy.set_xlim(x[0], x[-1])
#     ax_entropy.set_xlim(0, len(entropy_np))
    
#     # Match bottom plot limits to heatmap limits (tight, aligned)
#     # ax_entropy.set_xlim(ax_heatmap.get_xlim())          
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, f"{stay_id}_{event_type}_prediction_heatmap.png"), dpi=150, bbox_inches='tight')
#     plt.close()


def plot_individual_patient_predictions_old(        
        patient_data,
        plot_dir = None
    ):
    stay_id = patient_data['stay_id']
    event = patient_data['event']
    bin_edges = patient_data['time_bins']
    probs = patient_data['proabbilities']

    event_types = ['Survival', 'Death']
    event_type = event_types[event]

    plot_title = f"Patient ID: {stay_id} | Event: {event_type}"

    # 1. Reshape for heatmap: (16 bins) x (timesteps)
    # We stack Event 0 and Event 1 vertically
    timesteps = probs.shape[0]
    # Transpose to get (events, bins, timesteps) -> reshape to (16, timesteps)
    heatmap_data = probs.permute(1, 2, 0).reshape(16, -1).detach().cpu().numpy()

    # 2. Create Labels for the Y-axis
    # Formats as "Death [0-6h]", "Discharge [0-6h]", etc.
    y_labels = []
    for event in event_types:
        for i in range(len(bin_edges) - 1):
            y_labels.append(f"{event} [{bin_edges[i]}-{bin_edges[i+1]}h]")

    # 3. Plotting
    plt.figure(figsize=(12, 8))
    
    # Use LogNorm if your probabilities are very small (typical in ICU survival)
    # Use vmin/vmax to keep the colorbar consistent
    ax = sns.heatmap(
        heatmap_data, 
        xticklabels=False, # We will handle ticks manually for better control
        yticklabels=y_labels,
        cmap="YlGnBu", 
        norm=LogNorm(vmin=1e-4, vmax=heatmap_data.max()),
        cbar_kws={'label': 'Joint Probability $P(T, E)$'}
    )

    # Add a visual divider between Event 1 and Event 2
    plt.axhline(y=8, color='red', linewidth=2, linestyle='--')
    
    # 2. Get the number of timesteps
    num_timesteps = heatmap_data.shape[1]

    # 3. Manually configure the X-axis
    # MaxNLocator decides WHICH indices to label (e.g., [0, 50, 100...])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=12))

    # This Formatter ensures the label shown is the actual integer index (Hour)
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{int(x)}"
    
    ax.xaxis.set_major_formatter(major_formatter)

    # 4. Force the ticks to be visible (Seaborn sometimes turns them off)
    ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True, rotation=0)
    
    plt.title(plot_title)
    plt.xlabel("ICU Stay Duration (Hours)")
    plt.ylabel("Event & Forecast Horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{stay_id}_{event_type}_prediction_heatmap.png"))
    plt.close()
    
    


def main():
    predict()


if __name__ == "__main__":
    main()