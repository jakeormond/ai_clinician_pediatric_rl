"""
For generating predictions using a trained Dynamic-DeepHit model across the 
entire test dataset. Outputs predictions to a Parquet file along with metadata.

"""

# predict.py
import argparse
import json
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader

from ..data.data_processor_for_shards import ShardProcessor
from ..rl.replay_buffer import DynamicDeepHit_Lazy, worker_init_fn
from ..supervised_learning.models.dynamic_deephit import DynamicDeepHitMulti
from ..supervised_learning.models.dynamic_deephit_transformer import TransformerDeepHitMulti
from ..supervised_learning.training import create_processor_from_shards


def instantiate_model(config_manager, hyperparams, n_time_bins, num_risks, test_processor):
    feature_dim = test_processor.feature_dim
    if config_manager.model_type == "dynamic_deephit": # GRU-based Dynamic-DeepHit
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
    model_path = os.path.join(config_manager.experiment_dir, 'final_model.pt')
    device = config_manager.training_config['device']
    dynamic_deephit_model.load_state_dict(torch.load(
        model_path, 
        map_location=device,
        weights_only=True
    ))
    dynamic_deephit_model.eval()   
    return dynamic_deephit_model

def inference_loop(model, processor, config_manager, filename=None, device='cuda'):
    stay_ids = processor.dynamic_deephit_dataset['stay_ids']
    index_ptr = processor.dynamic_deephit_dataset['index_ptr']

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
            # (71 repeats of first step + actual trajectory)   
            pad = trajectory[0:1].repeat(71, 1)  # (71, F)
            padded_traj = torch.cat([pad, trajectory], dim=0)

            # 3. Create windows (T, 72, F)
            # unfold(dimension, size, step)
            windows = padded_traj.unfold(0, 72, 1).transpose(1, 2)
            
            # 4. Batch Inference
            batch_size = 64
            all_preds = []
            
            for chunk_start in range(0, T, batch_size):
                chunk_end = min(chunk_start + batch_size, T)
                windows_chunk = windows[chunk_start:chunk_end].to(device)
                
                risk_logits, _ = model(windows_chunk)

                # Extract the final prediction of each window in the chunk
                # and move back to CPU immediately to save GPU memory
                final_step_logits = [logits[:, -1, :].cpu() for logits in risk_logits]
                final_step_logits = torch.stack(final_step_logits, dim=2)  # (batch_size, num_risks, num_time_bins)
                
                T, D, K = final_step_logits.shape
                # reshape to (T, K, D)
                final_step_logits = final_step_logits.permute(0, 2, 1)

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
        file_path = os.path.join(config_manager.experiment_dir, filename + '.h5')
    else:
        file_path = os.path.join(config_manager.experiment_dir, 'test_set_predictions.h5')
    
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

    pass


def predict_all(config_manager): # for dynamic deephit models only
    models = ['GRU', 'transformer']
    cohorts = ['all_stays', 'sepsis_strict', 'sepsis_broad', 'ards_strict', 'ards_broad', 'aki_strict', 'aki_broad']

    for model_type in models:
        for cohort in cohorts:
            cohort_final_model_dir = os.path.join(
                config_manager.paths['final_model_dir'],
                model_type,
                cohort,
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
                continue
            else:
                # create DataLoader for test dataset
                # Create test_processor; note it will throw an error if the normalized test shard is missing.
                # All possible windows will be computed
                _, _, test_processor = create_processor_from_shards(config_manager, normalize=True)







def main():
    predict()


if __name__ == "__main__":
    main()