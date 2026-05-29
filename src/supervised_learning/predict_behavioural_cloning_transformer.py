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
        processor,
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

    if config_manager.training_config['actions_in_state']:
        action_ind = []
        for action in actions:
            ind = dataset.feature_names.index(action)
            action_ind.append(ind)

            mask_name = action + "_mask"
            if mask_name in dataset.feature_names:
                ind2 = dataset.feature_names.index(action + "_mask")
                action_ind.append(ind2)

            delta_name = action + "_delta"
            if delta_name in dataset.feature_names:
                ind3 = dataset.feature_names.index(action + "_delta")
                action_ind.append(ind3)

        response_zero_ind = np.sort(list(set(response_zero_ind) & set(action_ind)))
        carry_forward_ind = np.sort(list(set(carry_forward_ind) & set(action_ind)))  
         
    else: # response_zero_ind and carry_forward_ind should already be empty, in DynamicDeepHitTransformerDataset
        if len(response_zero_ind) > 0 or len(carry_forward_ind) > 0:
            raise ValueError("If actions are not included in state, then response_zero_ind and carry_forward_ind should be empty.")
    
    # BC transformer specific processing
    target_actions = dataset.target_actions

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
            trajectory = dataset.trajectories[start_idx:end_idx]
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
            processor,
            config_manager,
            hyperparams=hyperparams,
            filename=filename,
            model_type=model_type,
            experiment_dir=experiment_dir,
            device='cuda'
            )
        

    return dynamic_deephit_predictions 


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


def compare_models():
    models_dir = "/home/jake/Documents/AI_Clinician/models/bc_transformer/choc/sepsis/intake-vis/"
    with_actions_dir = os.path.join(models_dir, "with_actions", "final_model")
    without_actions_dir = os.path.join(models_dir, "without_actions", "final_model")

    # find the folder inside with_actions_dir and without_actions_dir
    with_actions_experiment_dir = os.path.join(with_actions_dir, os.listdir(with_actions_dir)[0])
    without_actions_experiment_dir = os.path.join(without_actions_dir, os.listdir(without_actions_dir)[0])

    predictions_file = "test_set_predictions.h5"
















def main():
    pass

if __name__ == "__main__":
    main()