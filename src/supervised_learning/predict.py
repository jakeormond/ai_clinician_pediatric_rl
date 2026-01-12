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

def inference_loop(model, processor, device='cuda'):
    stay_ids = processor.dynamic_deephit_dataset['stay_ids']
    index_ptr = processor.dynamic_deephit_dataset['index_ptr']
    
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
            pass
        
    pass  # Placeholder for potential future use



def predict(config_manager):

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
    inference_loop(dynamic_deephit_model, test_processor, device='cuda')

    













    
    test_dataset = DynamicDeepHit_Lazy(test_processor, config_manager)
    test_buffer = DataLoader(
        test_dataset,
        batch_size=config_manager.training_config['batch_size'],
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # create a model instance
    num_risks = 2
   
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

    # load trained model weights from final_model.pt
    model_path = os.path.join(config_manager.experiment_dir, 'final_model.pt')
    device = config_manager.training_config['device']
    dynamic_deephit_model.load_state_dict(torch.load(
        model_path, 
        map_location=device,
        weights_only=True
    ))
    dynamic_deephit_model.eval()   
    
    results = []
    with torch.no_grad():

        for idx, batch in enumerate(test_buffer):
            stay_ids = batch['stay_id']

            X = batch['features'].to(device)
            # time_to_event = batch[1].to(self.device)      # [batch_size]
            tte_raw = batch['tte_raw'].to(device)
            time_to_event = batch['time_to_event'].to(device)
            # event_label = batch[2].to(self.device)      # [batch_size]
            event_label = batch['event'].to(device)
            # mask = batch[3].to(self.device)
            mask = batch['mask'].to(device)

            logits, probs, _, _ , _ = dynamic_deephit_model(X, mask=mask, return_predictions=True)
                          
            results.append({
                'stay_id': stay_ids.cpu(),
                'tte_raw': tte_raw.cpu(),
                'time_to_event': time_to_event.cpu(),
                'event_label': event_label.cpu(),
                'logits': logits.cpu(),
                'probs': probs.cpu()
            })

    # convert time_bins to strings for column names
    time_bins_strings = []
    for i in range(len(time_bins)-1):
        time_bins_strings.append(f"bin{i}: {time_bins[i]}h-{time_bins[i+1]}h") 

    event_strings = ['discharge', 'mortality']
    all_rows = []
    for batch in results:
        stay_ids = batch['stay_id'].numpy()
        tte_raws = batch['tte_raw'].numpy()
        time_to_events = batch['time_to_event'].numpy()
        event_labels = batch['event_label'].numpy()
        # logits = batch['logits'].numpy()
        probs = batch['probs'].numpy()

        B, E, T = probs.shape

        for i in range(B):
            row = {
                'stay_id': stay_ids[i],
                'event_label': event_strings[event_labels[i]-1],
                'time_to_event(hours)': tte_raws[i],
                'time_to_event': time_to_events[i],
                
            }
            for e in range(E):
                for t in range(T):
                    row[f"probs_{event_strings[e]}_{time_bins_strings[t]}"] = probs[i, e, t]

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    # save as parquet
    output_path = os.path.join(config_manager.experiment_dir, 'test_set_predictions.parquet')
    df.to_parquet(output_path)
    print(f"Saved predictions to {output_path}")


def main():
    predict()


if __name__ == "__main__":
    main()