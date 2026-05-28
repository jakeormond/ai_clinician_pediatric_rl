"""
For generating predictions using a trained behavioural_cloning_transformer model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.
Based on predict.py which generates dynamic deephit transformer predictions

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
from ..supervised_learning.models.bc_transformer import BehaviouralCloneTransformer
from ..supervised_learning.training import create_processor_from_shards
from ..rl.replay_buffer import DynamicDeepHitTransformerDataset
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
        feature_dim
    ):
      
    bc_transformer_model = BehaviouralCloneTransformer(
        feature_dim=feature_dim + 1,   # input feature dimension - add 1 for the local time feature we add in the replay buffer
        d_model=hyperparams['d_model'],               # e.g., 128 or 256
        nhead=hyperparams['nhead'],                   # e.g., 4 or 8
        num_encoder_layers=hyperparams['n_layers'],   # e.g., 3 or 6
        ff_factor=hyperparams['ff_factor'],
        dropout=hyperparams['dropout'],
        feature_dropout_p=hyperparams['feature_dropout_p'],
        n_actions=len(hyperparams['actions']),
        max_seq_len=config_manager.training_config['window_length'],
    ).to(config_manager.training_config['device'])
    
    # load trained model weights from final_model.pt
    model_path = os.path.join(config_manager.experiment_dir, 'final_model.pt')
    device = config_manager.training_config['device']
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

    bc_transformer_model.load_state_dict(state_dict)
    
    bc_transformer_model.eval()   
    
    return bc_transformer_model

def inference_loop(
        model, 
        dataset, 
        config_manager, 
        hyperparams=None,
        filename=None, 
        model_type=None, 
        experiment_dir=None,
        device='cuda'
        ):
    
    stay_ids = dataset.stay_ids
    index_ptr = dataset.index_ptr
    window_length = config_manager.training_config['window_length']

    response_zero_ind = dataset.response_zero_ind
    carry_forward_ind = dataset.carry_forward_ind

    actions = config_manager.training_config['actions']
    
    if actions is None:
        # raise error
        raise ValueError("Actions must be specified in hyperparameters for behavioural cloning transformer inference.")

    action_ind = []
    targets_ind = []
    for action in actions:
        ind = processor.dynamic_deephit_dataset['feature_names'].index(action)
        action_ind.append(ind)
        targets_ind.append(ind)

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
    
    # BC transformer specific processing
    target_actions = torch.tensor(processor.dynamic_deephit_dataset["trajectories"][:, targets_ind], dtype=torch.float32)


    if ~config_manager.training_config['actions_in_state']:
        # drop columns from trajectories that correspond to response_zero_ind and carry_forward_ind since those features are not included in the state representation for the transformer
        cols_to_remove = np.sort(list(set(response_zero_ind) | set(carry_forward_ind)))
        processor.dynamic_deephit_dataset["trajectories"] = np.delete(processor.dynamic_deephit_dataset["trajectories"], cols_to_remove, axis=1)


    # create storage for predictions
    all_predictions = []   
    all_targets = []
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
            loss_mask = torch.ones((windows.shape[0], window_length), dtype=torch.bool, device=device)

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

                predicted_actions, output = model(windows_chunk, padding_mask=mask_chunk)
                # risk_logits = [logits[:, -1, :].cpu() for logits in risk_logits]  # unpack tuple
                # risk_logits = torch.stack(risk_logits, dim=2)  # (batch_size, num_risks, num_time_bins)

                # check if risk_logits has any nulls
                if torch.isnan(predicted_actions).any():
                    raise ValueError(f"NaN values found in risk_logits for stay_id {stay_id} at chunk {chunk_start}-{chunk_end}")

                # check if output has any nulls
                if torch.isnan(output).any():
                    raise ValueError(f"NaN values found in model output for stay_id {stay_id} at chunk {chunk_start}-{chunk_end}")

                # reshape to (T, K, D)
                # final_step_logits = risk_logits.permute(0, 2, 1)

                # add to list of all predictions    
                all_preds.append(predicted_actions)
                all_outs.append(output.cpu())
            
            # 5. concatenate all predictions for this patient
            patient_preds = torch.cat(all_preds, dim=0)  
            all_predictions.append(patient_preds)
            all_targets.append(target_actions[start_idx:end_idx].cpu())
            all_outputs.append(torch.cat(all_outs, dim=0))    
       

    # concatenate all patient predictions into a numpy array
    all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
    all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
   
    # add predictions to processor
    processor.dynamic_deephit_dataset['predictions'] = all_predictions
    processor.dynamic_deephit_dataset['model_outputs'] = all_outputs
    processor.dynamic_deephit_dataset['targets'] = all_targets

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
        hyperparams = results_dict['hyperparameters']    
             
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        _, _, test_processor = create_processor_from_shards(config_manager, normalize=True)

        test_dataset = DynamicDeepHitTransformerDataset(test_processor, config_manager)
        feature_dim = test_dataset.trajectories.shape[1] # the model adds + 1 for the local time feature that gets added
        # instantiate the model
        bc_transformer_model = instantiate_model(config_manager, hyperparams, feature_dim)

        # run inference on test set - SWAP IN test_dataset!!!!!!!!!
        dynamic_deephit_dataset = inference_loop(bc_transformer_model, test_processor, config_manager,device='cuda')



            

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
    filename = f'{set}_set_predictions'
    filepath = os.path.join(experiment_dir, filename + '.h5')
    if os.path.exists(filepath):
        # load existing predictions
        print(f"Loading existing predictions from {filepath}")
        keys_to_load = ['stay_ids', 'event', 'predictions', 'targets', 'model_outputs']
        dynamic_deephit_predictions = ShardProcessor.load_dynamic_deephit_predictions_from_h5(filepath, keys_to_load = keys_to_load)
    else:
        # create DataLoader for test dataset
        # Create test_processor; note it will throw an error if the normalized test shard is missing.
        # All possible windows will be computed
        processor = create_processor_from_shards(config_manager, normalize=True, set_to_return = set)
        dataset = DynamicDeepHitTransformerDataset(processor, config_manager)

        # instantiate the model
        # load result.json from config_manager.experiment_dir to get hyperparameters
        with open(
            os.path.join(experiment_dir, 'results.json'), 'r'
        ) as f:
            results_dict = json.load(f)
        hyperparams = results_dict['hyperparameters']    
        feature_dim = dataset.trajectories.shape[1]


        bc_transformer_model = instantiate_model(config_manager, hyperparams, feature_dim)

        # run inference on test set - this function also saves the predictions to file
        filename = f'{set}_set_predictions'
        dynamic_deephit_predictions = inference_loop(
            bc_transformer_model, 
            dataset, 
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
    
    
    

def get_formatted_cell(boot_samples):
    """
    Takes a list of bootstrap scores and returns a 'Mean (CI_Low - CI_High)' string.
    """
    mean = np.mean(boot_samples)
    low = np.percentile(boot_samples, 2.5)
    high = np.percentile(boot_samples, 97.5)
    
    # Using 3 decimal places is standard for clinical journals
    return f"{mean:.3f} ({low:.3f}–{high:.3f})"




def plot_random_predictions(
        config_manager,
        model='bc_transformer',
        cohort='sepsis',
    ):
    
    plot_dir = os.path.join(config_manager.experiment_dir, 'prediction_plots')
    os.makedirs(plot_dir, exist_ok=True)

    filename = 'test_set_predictions'
    filepath = os.path.join(config_manager.experiment_dir, filename + '.h5')
    # load existing predictions
    print(f"Loading existing predictions from {filepath}")
    keys_to_load = ['stay_ids', 'event', 'predictions', 'targets', 'index_ptr']
    dynamic_deephit_predictions = ShardProcessor.load_dynamic_deephit_predictions_from_h5(filepath,keys_to_load=keys_to_load)
    
    # loading the dataset to get trajectories
    # dataset_filepath = os.path.join(config_manager.get_data_dir, config_manager.data_config['shard_dir'], 
    #             f"split_{config_manager.split_num}", "training_set", "test_dynamic_deephit_dataset.h5")    
    # dynamic_deephit_dataset = ShardProcessor.load_dynamic_deephit_dataset_from_h5(dataset_filepath)
    
    # confirm that the predictions and dataset are aligned by index_ptr
    # if not np.array_equal(dynamic_deephit_predictions['index_ptr'], dynamic_deephit_dataset['index_ptr']):
    #     raise ValueError("Index pointers in predictions and dataset do not match. Cannot plot predictions.")
    # else:
    #     print("Index pointers in predictions and dataset match. Proceeding with plotting.")

    # check that trajectories and predictions have same number of rows
    # if dynamic_deephit_predictions['predictions'].shape[0] != dynamic_deephit_dataset['trajectories'].shape[0]:
    #     raise ValueError("Number of prediction rows does not match number of trajectory rows. Cannot plot predictions.")
    # else:
    #     print("Number of prediction rows matches number of trajectory rows. Proceeding with plotting.")
    
    
    # plot predictions for a random sample of 10 patients
    stay_ids = dynamic_deephit_predictions['stay_ids']
    events = dynamic_deephit_predictions['event']

    death_events = [i for i, e in enumerate(events) if e == 1]
    survival_events = [i for i, e in enumerate(events) if e == 0]

    # randomly sample 50 of each event type
    np.random.seed(42)
    n_samples_per_event = 50
    if len(death_events) < n_samples_per_event:
        sampled_death_indices = death_events
    else:    
        sampled_death_indices = np.random.choice(death_events, size=n_samples_per_event, replace=False)
    sampled_survival_indices = np.random.choice(survival_events, size=n_samples_per_event, replace=False)

    sampled_indices = np.concatenate([sampled_death_indices, sampled_survival_indices])

    for idx in sampled_indices:
        stay_id = stay_ids[idx]
        event = dynamic_deephit_predictions['event'][idx]

        start_idx = dynamic_deephit_predictions['index_ptr'][idx]
        end_idx = dynamic_deephit_predictions['index_ptr'][idx + 1]

        predictions = dynamic_deephit_predictions['predictions'][start_idx:end_idx]
        targets = dynamic_deephit_predictions['targets'][start_idx:end_idx]
        # probabilities = calculate_probabilities(predictions)

        # entropy = dynamic_deephit_predictions['entropy'][start_idx:end_idx]
        # health_score = dynamic_deephit_predictions['health_score'][start_idx:end_idx]

        # patient_data = {
        #     'stay_id': stay_id,
        #     'event': event,
        #     'time_bins': config_manager.time_bins,
        #     'probabilities': probabilities,
        #     'entropy': entropy,
        #     'health_score': health_score
        # }   

        patient_data = {
            'stay_id': stay_id,
            'event': event,
            'predictions': predictions,
            'targets': targets
        }   


        plot_individual_patient_predictions(
            patient_data,
            config_manager,
            plot_dir = plot_dir
        )
        print(f"Prediction plot for patient {stay_id} saved to {plot_dir}")



def plot_individual_patient_predictions(        
        patient_data,
        config_manager,
        plot_dir = None
    ):
    stay_id = patient_data['stay_id']
    event = patient_data['event']
    predictions = patient_data['predictions']
    targets = patient_data['targets']

    event_types = ['Survival', 'Death']
    event_type = event_types[event]

    plot_title = f"Patient ID: {stay_id} | Event: {event_type}"

    actions = config_manager.training_config['actions']
    
    fig = plt.figure(figsize=(14, 10))

    for i, action in enumerate(actions):
        # create subplot for this action
        ax = fig.add_subplot(len(actions), 1, i + 1)
        # plot predictions[:, i] as a line plot
        ax.plot(predictions[:, i], label='Predicted action', color='blue', linewidth=2)
        # plot targets[:, i] as a line plot
        ax.plot(targets[:, i], label='Clinician action', color='orange', linewidth=2, linestyle='--')
        ax.set_title(f"Action: {action}", fontsize=11, fontweight='bold')
        ax.set_xlabel('ICU Stay Duration (Hours)', fontsize=10)
        ax.set_ylabel('log-min-max', fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)



    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{stay_id}_{event_type}_prediction_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()




    


def main():
    predict()


if __name__ == "__main__":
    main()