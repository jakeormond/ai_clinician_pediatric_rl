import argparse
import json
import os
from typing_extensions import final

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats as scipy_stats
from pathlib import Path
import pickle
from .training import create_buffer
from .models.shared_components import TrainerUtils
from ..setup_environment import load_paths

# --- comparison of SARSA critics (new-style analysis; uses the runs / models / data infrastructure) ---
import warnings
from typing import Any, Dict
from src.rl.analysis import data as D
from src.rl.evaluate_cql_alpha_sweep import actor_actions, check_judge_matches
from src.rl.evaluate_fqe_final_seeds import _by_seed, _pct_ci, mc_calibration_one_seed



RL_MODELS_PATH = "/home/jake/Documents/AI_Clinician/models/rl_models/choc" 


def _load_npz_scalar(npz_data, key):
    """Safely extract a scalar from an NPZ entry, normalizing NaN to None."""
    value = npz_data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def load_sarsa_critic(processor, hyperparams, model_path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_dim = processor.rl_dataset["states"].shape[1]
    action_dim = processor.rl_dataset["actions"].shape[1]

    sarsa_model = Critic(
        state_dim, 
        action_dim, 
        hidden_dim=hyperparams['hidden_dim'], 
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sarsa_model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['critic_state_dict']
    sarsa_model.load_state_dict(state_dict)
    sarsa_model.eval()

    return sarsa_model



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models on SARSA CHOC")

    # Keyword Arguments (Match your launch.json)
    parser.add_argument("--hospital", type=str, default="choc")
    parser.add_argument("--experiment_type", type=str, default="reinforcement_learning")
    parser.add_argument("--experiment_variant", type=str, default="hyperparameter_search_evaluation")
    parser.add_argument("--model_type", type=str, default="basic_q")
    parser.add_argument("--mode", type=str, default="single_split")
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="sepsis")
    parser.add_argument("--actions", type=str, default="vis-intake")
    parser.add_argument("--health_score_type", type=str, default=None)
    parser.add_argument("--discount", type=float, default=None)
    
    # The Flag (Action: store_true means if it's in the args, it becomes True)
    parser.add_argument("--testing", action="store_true", help="Run in test mode")

    return parser.parse_args()


def load_configuration(config_manager_path):
    if not os.path.exists(config_manager_path):
        print(f"Config manager not found in: {config_manager_path}, skipping evaluation.")
        raise FileNotFoundError(f"Config manager not found in: {config_manager_path}")
    else:
        print(f"Loading config manager from: {config_manager_path}")
        with open(config_manager_path, "rb") as cfg_file:
            config_manager = pickle.load(cfg_file)
    
    # set paths 
    config_manager.paths, config_manager.environment = load_paths(config_manager.hospital)
    if config_manager.environment == "local":
        config_manager.config_dir = Path(config_manager.paths['config_dir'])
    config_manager.set_data()
    config_manager.reset_cohort(config_manager.cohort)

    return config_manager


def load_hyperparameters(results_path):
    with open(results_path, "r") as file:
        results = json.load(file)
    hyperparams = results["hyperparameters"]

    return hyperparams


def create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams):
    action_names=config_manager.data_config['actions']
    n_step = hyperparams['n_step']
    gamma = hyperparams['discount']
    use_compressed_latents = hyperparams['use_compressed_latents']

    scaler = train_processor.create_rl_dataset(
        states="model_outputs", 
        scaler=None, 
        action_names=action_names, 
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma
    ) 

    _ = test_processor.create_rl_dataset(
        states="model_outputs",
        scaler=scaler,
        action_names=action_names,
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma
    )

    return train_processor, test_processor, scaler


def get_trajectory_data(processor, traj_start_inds, stay_id, as_tensor=True):

    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]
    # get index of stay_id in stay_ids
    stay_idx = np.where(stay_ids == stay_id)[0][0]
    indices = np.arange(traj_start_inds[stay_idx], traj_start_inds[stay_idx+1])
    states = processor.rl_dataset["states"][indices]
    actions = processor.rl_dataset["actions"][indices]
    rewards = processor.rl_dataset["rewards"][indices]
    next_states = processor.rl_dataset["next_states"][indices]
    next_actions = processor.rl_dataset["next_actions"][indices]
    dones = processor.rl_dataset["dones"][indices]
    mortalities = processor.rl_dataset["mortality"][indices]

    if not as_tensor:
        return states, actions, rewards, next_states, next_actions, dones, mortalities, indices

    # convert to torch tensors
    states = torch.as_tensor(states, dtype=torch.float32).to('cuda')
    actions = torch.as_tensor(actions, dtype=torch.float32).to('cuda')
    rewards = torch.as_tensor(rewards, dtype=torch.float32).to('cuda')
    next_states = torch.as_tensor(next_states, dtype=torch.float32).to('cuda')
    next_actions = torch.as_tensor(next_actions, dtype=torch.float32).to('cuda')
    dones = torch.as_tensor(dones, dtype=torch.float32).to('cuda')
    mortalities = torch.as_tensor(mortalities, dtype=torch.float32).to('cuda')

    return states, actions, rewards, next_states, next_actions, dones, mortalities, indices



@torch.no_grad()
def hyperparameter_search_evaluation(args=None):

    discount = args.discount
    
    search_results = []
    
    models_dir = os.path.join(RL_MODELS_PATH, 'basic_q', 
        f"split_{args.split_num}", args.cohort, args.actions)

    # get the model directories
    search_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    # exclude folder raw_health_score_only 
    search_folders = [f for f in search_folders if f != "raw_health_score_only"]
    
    search_num = 0
    
    for f in search_folders:
        # load the config_manager.pkl file from the model folder
        config_manager_path = os.path.join(models_dir, f, "config_manager.pkl")
        config_manager = load_configuration(config_manager_path)

        if search_num == 0:
            # load data
            train_processor, val_processor, _ = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')
        
        inner_folders = [inner_f for inner_f in os.listdir(os.path.join(models_dir, f)) 
                         if os.path.isdir(os.path.join(models_dir, f, inner_f))]
        
        for inner_f in inner_folders:
            model_path = os.path.join(models_dir, f, inner_f, "final_model.pt")
            
            print (f"Evaluating model: {model_path}")

            print(f"Evaluating model in folder: {model_path}")
        
            # load hyperprameters from results.json
            results_path = os.path.join(models_dir, f, inner_f, "results.json")
            hyperparams = load_hyperparameters(results_path)

            if hyperparams['discount'] != discount:
                print(f"Skipping model {model_path} due to discount mismatch: {hyperparams['discount']} != {discount}")
                continue
                                 
            # process the reward
            health_score_median = train_processor.process_reward(config_manager, hyperparams)
            val_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
            
            # Create replay buffers - THESE CALLS CREATE THE RL DATASET IN THE PROCESSORS!!!!!!!!!!!!!!!!!!!!!!!!
            batch_size = 1024
            training_buffer, scaler = create_buffer(
                train_processor, 
                batch_size, 
                hyperparams,
                action_names=config_manager.data_config['actions'],
            )
   
            # create val buffer
            val_buffer, _ = create_buffer(
                val_processor,
                batch_size, 
                hyperparams,
                scaler=scaler, 
                action_names=config_manager.data_config['actions'],
                shuffle=False
            )
                        
            sarsa_model = load_sarsa_critic(val_processor, hyperparams, model_path=model_path)
           
            sarsa_model.eval()        
    
            q1_values = []
            q2_values = []
            target_q_values = []
            mortality_values = []
            for batch_idx, batch in enumerate(val_buffer):
                state, action, reward, next_state, next_action, done, mortality = TrainerUtils._unpack_batch(batch, 'cuda')
                not_done = 1 - done
                mortality_values.append(mortality.detach().cpu().numpy())

                q1, q2 = sarsa_model(state, action)
                
                # Compute the target Q value
                target_Q1, target_Q2 = sarsa_model(next_state, next_action)
                target_Q = (target_Q1 + target_Q2) / 2.0
                # target_Q = torch.min(target_Q1, target_Q2)  # TD3 style: take the minimum of the two critics to mitigate overestimation bias
                target_Q = reward + not_done * hyperparams['discount'] * target_Q	

                q1_values.append(q1.detach())
                q2_values.append(q2.detach())
                target_q_values.append(target_Q.detach())

            q1_values = torch.cat(q1_values, dim=0)
            q2_values = torch.cat(q2_values, dim=0)
            target_q_values = torch.cat(target_q_values, dim=0)

            current_q_values = (q1_values + q2_values) / 2.0

            # bellman residual calculation
            bellman_residual = F.mse_loss(current_q_values, target_q_values)
            normalized_bellman_residual = bellman_residual / (target_Q.var() + 1e-8) 

            # Q1-Q2 uncertainty calculation
            disagreement = F.mse_loss(q1_values, q2_values)
            mean_Q   = (q1_values + q2_values) / 2.0
            q_var    = mean_Q.var()
            normalized_uncertainty = (disagreement / (q_var + 1e-8)).item()

            # calculate auc for current_q_values and mortality_values
            mortality_values = np.concatenate(mortality_values, axis=0)
            y_true = 1 - mortality_values  # Assuming mortality=1 means death, we want to predict survival
            y_score = current_q_values.detach().cpu().numpy().ravel()         
            auc = roc_auc_score(y_true, y_score)
            print(f"Model: {model_path}, AUC: {auc:.4f}")

            search_results.append({
                "model_path": model_path,
                "discount": hyperparams['discount'],
                "learning_rate": hyperparams['learning_rate'],
                "tau": hyperparams['tau'],
                "hidden_dim": hyperparams['hidden_dim'],
                "bellman_residual": bellman_residual.item(),
                "normalized_bellman_residual": normalized_bellman_residual.item(),
                "disagreement": disagreement.item(),
                "normalized_uncertainty": normalized_uncertainty,
                "auc": auc
            })

            search_num += 1
        
    # concatenate search_results into a dataframe and save to csv
    df = pd.DataFrame(search_results)
    # reorder rows by normalized_bellman_residual ascending
    df = df.sort_values(by="normalized_bellman_residual", ascending=True)
    df.to_csv(os.path.join(models_dir, f"hyperparameter_search_evaluation_results_{discount}.csv"), index=False)    

@torch.no_grad()
def evaluate_final_model(args=None):
    discount = args.discount
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = os.path.join(RL_MODELS_PATH, 'basic_q', 
        f"split_{args.split_num}", args.cohort, args.actions, 
        args.health_score_type, f"final_model_discount_{discount}")
    # get the model dir inside the final_model folder
    inner_folders = [inner_f for inner_f in os.listdir(model_dir) 
                     if os.path.isdir(os.path.join(model_dir, inner_f))]
    if len(inner_folders) == 0:
        print(f"No model found in folder: {model_dir}")
        return
    elif len(inner_folders) > 1:
        print(f"Multiple models found in folder: {model_dir}, evaluating the first one: {inner_folders[0]}")
        return
    
    # check if model output files already exist, if so, skip evaluation
    save_dir = os.path.join(model_dir, inner_folders[0], "sarsa_analysis")
    if os.path.exists(os.path.join(save_dir, "action_analysis.npz")):
        print(f"Model output files already exist in: {save_dir}, loading...")
        
        # load the action_analysis.npz file and print the keys
        action_analysis = np.load(os.path.join(save_dir, "action_analysis.npz"), allow_pickle=True)
        print(f"Loaded action_analysis.npz, keys: {action_analysis.files}")
        return action_analysis, save_dir

       
    model_path = os.path.join(model_dir, inner_folders[0], "final_model.pt")
    print(f"Evaluating final model: {model_path}")

    config_manager_path = os.path.join(model_dir, inner_folders[0], "config_manager.pkl")
    config_manager = load_configuration(config_manager_path)

    results_path = os.path.join(model_dir, inner_folders[0], "results.json")
    hyperparams = load_hyperparameters(results_path)

    train_processor, _, test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, hyperparams)
    test_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)

    # create rl dataset
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams)

    # load the model
    sarsa_model = load_sarsa_critic(test_processor, hyperparams, model_path=model_path)

    # get the actions and get their standard deviation
    actions = test_processor.rl_dataset["actions"]
    shuffled_actions = actions[torch.randperm(actions.shape[0])]
    action_dim = actions.shape[1]
    
    action_std = torch.as_tensor(actions.std(axis=0), dtype=torch.float32).to(device)
    print(f"Action std: {action_std}")
    
    # loop through trajectories in test_processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(test_processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = test_processor.dynamic_deephit_dataset["stay_ids"]
    
    batch_size = 1024
    K = 50

    # storage for q_values, mortalities, and headroom for each stay
    q_values        = []
    mortality_values = []
    headroom_joint  = []      # best plausible alt (both dims perturbed) - clinician
    sensitivity_joint = []    # std of Q across K joint perturbations
    noise_headroom = []
    disagreement    = []      # |Q1 - Q2| at clinician action
    headroom_per_dim = []     # (N, action_dim): best single-dim alt - clinician
    direction_per_dim = []    # (N, action_dim): signed (best_alt_action - clinician_action) per dim
    temporal_index  = [] 
    
    for stay_id in stay_ids:

        states, actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=True) 
        shuffled_actions_stay = shuffled_actions[indices]

        n_batches = int(np.ceil(states.shape[0] / batch_size))
        
        # storage for q_values, mortalities, and headroom for each batch
        q_for_stay        = []
        mort_for_stay     = mortalities
        hr_joint_stay     = []
        sens_joint_stay   = []
        noise_headroom_stay = []
        disagree_stay     = []
        hr_dim_stay       = []
        dir_dim_stay      = []     # position within trajectory

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, states.shape[0])
            states_batch = states[start_idx:end_idx]
            actions_batch = actions[start_idx:end_idx]
            shuffled_actions_batch = (shuffled_actions_stay[start_idx:end_idx]).to(device)

            # evaluate the model on the batch
            with torch.no_grad():
                q1_clin, q2_clin = sarsa_model(states_batch, actions_batch)
                q_clin = (q1_clin + q2_clin) / 2.0
                disagree = (q1_clin - q2_clin).abs()

                q1_shuffle, q2_shuffle = sarsa_model(states_batch, shuffled_actions_batch)
                q_shuffle = (q1_shuffle + q2_shuffle) / 2.0

                # --- joint perturbation: both dims at once ---
                best_joint = torch.full_like(q_clin, float('-inf'))
                best_shuffled = torch.full_like(q_clin, float('-inf'))
                joint_q_samples = []

                for _ in range(K):
                    noise = torch.randn_like(actions_batch) * action_std
                    alt   = (actions_batch + noise).clamp(0.0, 1.0)
                    q1_a, q2_a = sarsa_model(states_batch, alt)
                    q_a = (q1_a + q2_a) / 2.0
                    best_joint = torch.maximum(best_joint, q_a)
                    joint_q_samples.append(q_a)

                    alt_shuffled = (shuffled_actions_batch + noise).clamp(0.0, 1.0)  # use the shuffled actions for the alternative
                    q1_a_shuffled, q2_a_shuffled = sarsa_model(states_batch, alt_shuffled)
                    q_a_shuffled = (q1_a_shuffled + q2_a_shuffled) / 2.0
                    best_shuffled = torch.maximum(best_shuffled, q_a_shuffled)
                
                joint_q_samples = torch.cat(joint_q_samples, dim=1)   # (B, K)
                hr_joint   = best_joint - q_clin
                sens_joint = joint_q_samples.std(dim=1, keepdim=True)  # (B,1)

                noise_headroom_batch = best_shuffled - q_shuffle
                
                
                # --- per-dimension perturbation: vary one dim, hold others at clinician value ---
                best_dim_q   = torch.full((states_batch.shape[0], action_dim),
                                        float('-inf'), device=device)
                best_dim_act = actions_batch.clone()  # will overwrite per dim with best al

                for d in range(action_dim):
                    best_q_d   = torch.full((states_batch.shape[0], 1), float('-inf'), device=device)
                    best_act_d = actions_batch[:, d:d+1].clone()
                    for _ in range(K):
                        alt = actions_batch.clone()
                        noise_d = torch.randn(states_batch.shape[0], 1, device=device) * action_std[d]
                        alt[:, d:d+1] = (actions_batch[:, d:d+1] + noise_d).clamp(0.0, 1.0)
                        q1_a, q2_a = sarsa_model(states_batch, alt)
                        q_a = (q1_a + q2_a) / 2.0           # (B,1)
                        improved = q_a > best_q_d
                        best_q_d   = torch.where(improved, q_a, best_q_d)
                        best_act_d = torch.where(improved, alt[:, d:d+1], best_act_d)
                    best_dim_q[:, d:d+1]   = best_q_d
                    best_dim_act[:, d:d+1] = best_act_d
                
                hr_dim  = best_dim_q - q_clin                       # (B, action_dim)
                dir_dim = best_dim_act - actions_batch              # (B, action_dim) signed
            
            q_for_stay.append(q_clin)
            hr_joint_stay.append(hr_joint)
            sens_joint_stay.append(sens_joint)
            noise_headroom_stay.append(noise_headroom_batch)
            disagree_stay.append(disagree)
            hr_dim_stay.append(hr_dim)
            dir_dim_stay.append(dir_dim)
            
        # concatenate this stay, record temporal index within the stay
        n_t = sum(x.shape[0] for x in q_for_stay)
        q_values.append(torch.cat(q_for_stay).cpu().numpy())
        mortality_values.append(mort_for_stay.cpu().numpy())
        headroom_joint.append(torch.cat(hr_joint_stay).cpu().numpy())
        sensitivity_joint.append(torch.cat(sens_joint_stay).cpu().numpy())
        disagreement.append(torch.cat(disagree_stay).cpu().numpy())
        noise_headroom.append(torch.cat(noise_headroom_stay).cpu().numpy())
        headroom_per_dim.append(torch.cat(hr_dim_stay).cpu().numpy())
        direction_per_dim.append(torch.cat(dir_dim_stay).cpu().numpy())
        temporal_index.append(np.arange(n_t))
    
    # concatenate across stays
    q_values          = np.concatenate(q_values).ravel()
    mortality_values  = np.concatenate(mortality_values).ravel()
    headroom_joint    = np.concatenate(headroom_joint).ravel()
    sensitivity_joint = np.concatenate(sensitivity_joint).ravel()
    disagreement      = np.concatenate(disagreement).ravel()
    noise_headroom    = np.concatenate(noise_headroom).ravel()
    headroom_per_dim  = np.concatenate(headroom_per_dim)    # (N, action_dim)
    direction_per_dim = np.concatenate(direction_per_dim)   # (N, action_dim)
    temporal_index    = np.concatenate(temporal_index).ravel()
    clinician_actions  = test_processor.rl_dataset["actions"].cpu().numpy()

    threshold, thresholds, real_ccdf, noise_ccdf, gap = find_calibrated_threshold(headroom_joint, noise_headroom, min_relative_gap=0.2, n_thresholds=500, stable_window=40)

    # calculate auc for q_values and mortality_values
    y_true = 1 - mortality_values  # Assuming mortality=1 means death, we want to predict survival
    y_score = q_values.ravel()
    roc_auc = roc_auc_score(y_true, y_score)
    print(f"Final model AUC: {roc_auc:.4f}")

    # calculate auprc for q_values and mortality_values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    print(f"Final model AUPRC: {auprc:.4f}")


    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(save_dir, "action_analysis.npz"),
        traj_start_inds=traj_start_inds,
        stay_ids=stay_ids,
        q_values=q_values,
        mortality_values=mortality_values,
        headroom_joint=headroom_joint,
        sensitivity_joint=sensitivity_joint,
        noise_headroom=noise_headroom,
        disagreement=disagreement,
        headroom_per_dim=headroom_per_dim,        # (N, action_dim)
        direction_per_dim=direction_per_dim,      # (N, action_dim)
        temporal_index=temporal_index,
        clinician_actions=clinician_actions,
        threshold=np.nan if threshold is None else float(threshold),
        thresholds=thresholds,
        real_ccdf=real_ccdf,
        noise_ccdf=noise_ccdf,
        gap=gap,
        roc_auc=roc_auc,
        auprc=auprc
    )

    action_analysis = np.load(os.path.join(save_dir, "action_analysis.npz"), allow_pickle=True)
    print(f"Saved action_analysis.npz, keys: {action_analysis.files}")

    return action_analysis, save_dir



def find_calibrated_threshold(real_headroom, noise_headroom,
                                         min_relative_gap=0.5, n_thresholds=500,
                                         stable_window=20):
    """
    Find the threshold where the real headroom CCDF exceeds the noise-floor
    CCDF by at least `min_relative_gap` as a RELATIVE (not absolute) amount.
    This is robust on a log-scale CCDF, where absolute gaps shrink as you
    move into the tail even when the relative separation is large and real.

    min_relative_gap: e.g. 0.5 means "real curve is at least 50% higher
                       than noise curve at this threshold"
    require the gap to hold for `stable_window` consecutive threshold
    steps, so a single noisy crossing isn't mistaken for real divergence
    """
    max_val = max(real_headroom.max(), noise_headroom.max())
    thresholds = np.linspace(0, max_val, n_thresholds)

    real_ccdf  = np.array([(real_headroom  > t).mean() for t in thresholds])
    noise_ccdf = np.array([(noise_headroom > t).mean() for t in thresholds])

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_excess = np.where(
            noise_ccdf > 0,
            (real_ccdf - noise_ccdf) / noise_ccdf,
            np.where(real_ccdf > 0, np.inf, 0.0)
        )

    candidate_idx = np.where(relative_excess >= min_relative_gap)[0]

    if len(candidate_idx) == 0:
        return None, thresholds, real_ccdf, noise_ccdf, relative_excess

    for idx in candidate_idx:
        window_end = min(idx + stable_window, len(relative_excess))
        if window_end - idx == stable_window and \
           (relative_excess[idx:window_end] >= min_relative_gap).all():
            return thresholds[idx], thresholds, real_ccdf, noise_ccdf, relative_excess

    # fall back to first crossing if no fully stable window found
    threshold = thresholds[candidate_idx[0]]
    return threshold, thresholds, real_ccdf, noise_ccdf, relative_excess



def plot_calibration(q_values, mortality_values, survivor, nonsurvivor, 
                     n_bins=20, min_bin_size=50, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1a. Q distribution by outcome
    axes[0].hist(q_values[survivor],    bins=60, alpha=0.6, 
                 label='Survivors',     density=True)
    axes[0].hist(q_values[nonsurvivor], bins=60, alpha=0.6, 
                 label='Non-survivors', density=True)
    axes[0].set_xlabel('Clinician Q-value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Q separates outcomes')
    axes[0].legend()

    # 1b. Calibration curve with confidence intervals
    bin_edges = np.quantile(q_values, np.linspace(0, 1, n_bins + 1))
    bin_idx   = np.clip(np.digitize(q_values, bin_edges[1:-1]), 0, n_bins - 1)

    bin_centres, surv_rate, ci_low, ci_high, bin_ns = [], [], [], [], []

    for b in range(n_bins):
        m = bin_idx == b
        n = m.sum()
        if n < min_bin_size:
            continue
        
        p = (mortality_values[m] == 0).mean()   # survival rate
        
        # Wilson confidence interval — handles proportions near 0 or 1 better than normal approx
        z    = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        
        bin_centres.append(q_values[m].mean())
        surv_rate.append(p)
        ci_low.append(max(0, centre - half))
        ci_high.append(min(1, centre + half))
        bin_ns.append(n)

    bin_centres = np.array(bin_centres)
    surv_rate   = np.array(surv_rate)
    ci_low      = np.array(ci_low)
    ci_high     = np.array(ci_high)

    axes[1].plot(bin_centres, surv_rate, 'o-', lw=2, zorder=3)
    axes[1].fill_between(bin_centres, ci_low, ci_high, alpha=0.2, label='95% CI')
    
    # annotate bin sizes so you can see where estimates are reliable
    for x, y, n in zip(bin_centres, surv_rate, bin_ns):
        axes[1].annotate(f'n={n}', (x, y), textcoords='offset points',
                         xytext=(0, 6), ha='center', fontsize=7, color='gray')

    overall_surv = (mortality_values == 0).mean()
    axes[1].axhline(0.5, color='gray', ls=':', lw=1, label='50% survival')
    axes[1].axhline(overall_surv, color='green', ls='--', lw=1, 
                    label=f'Population average ({overall_surv:.2f})')
    
    # plateau_threshold = bin_centres[4]  # approximately the 7th bin centre

    # axes[1].axvspan(bin_centres[0], plateau_threshold, 
    #                 alpha=0.08, color='red', 
    #                 label='Discriminative region')
    # axes[1].axvspan(plateau_threshold, bin_centres[-1], 
    #                 alpha=0.08, color='green', 
    #                 label='Plateau region')


    axes[1].set_xlabel('Mean Q-value in bin')
    axes[1].set_ylabel('Realised survival rate')
    axes[1].set_title(f'Higher Q → better realised survival ({n_bins} bins)')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "calibration_plot.png"), 
                    dpi=150, bbox_inches='tight')
    return fig, axes



def plot_headroom(headroom_joint, trust_mask, survivor, nonsurvivor,
                   calibrated_threshold, save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    hr = headroom_joint[trust_mask]
    hr_sorted = np.sort(hr)
    ccdf = 1.0 - np.arange(len(hr_sorted)) / len(hr_sorted)
    axes[0].plot(hr_sorted, ccdf, lw=2)
    axes[0].axvline(calibrated_threshold, color='k', ls=':', lw=1.5,
                    label=f'Calibrated threshold = {calibrated_threshold:.4f}')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Headroom threshold')
    axes[0].set_ylabel('Fraction of states with headroom > threshold')
    pct_improvable = (hr > calibrated_threshold).mean() * 100
    axes[0].set_title(f'{pct_improvable:.2f}% of states genuinely improvable')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both', alpha=0.3)

    for label, mask, color in [('Survivors',     survivor & trust_mask,    'tab:blue'),
                               ('Non-survivors', nonsurvivor & trust_mask, 'tab:orange')]:
        h = np.sort(headroom_joint[mask])
        c = 1.0 - np.arange(len(h)) / len(h)
        axes[1].plot(h, c, lw=2, label=label, color=color)
    axes[1].axvline(calibrated_threshold, color='k', ls=':', lw=1.5)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Headroom threshold')
    axes[1].set_ylabel('Fraction of states with headroom > threshold')
    axes[1].set_title('Headroom tail: survivors vs non-survivors')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "headroom_plot.png"), dpi=150, bbox_inches='tight')
    return fig, axes


def plot_temporal_structure(sensitivity_joint, temporal_index, trust_mask, 
                  max_hours=None, min_n=20, save_dir=None):
    
    # determine max hours from data if not specified
    if max_hours is None:
        max_hours = int(temporal_index.max()) + 1

    hours, mean_sens, sem_sens, n_patients = [], [], [], []
    
    for t in range(max_hours):
        m = (temporal_index == t) & trust_mask
        n = m.sum()
        if n >= min_n:
            hours.append(t)
            mean_sens.append(sensitivity_joint[m].mean())
            sem_sens.append(sensitivity_joint[m].std() / np.sqrt(n))
            n_patients.append(n)

    hours     = np.array(hours)
    mean_sens = np.array(mean_sens)
    sem_sens  = np.array(sem_sens)
    n_patients = np.array(n_patients)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # primary: sensitivity curve with SEM ribbon
    ax1.plot(hours, mean_sens, lw=2, color='tab:blue', 
             label='Action sensitivity')
    ax1.fill_between(hours, mean_sens - sem_sens, mean_sens + sem_sens, 
                     alpha=0.3, color='tab:blue')
    ax1.set_xlabel('Hours since ICU admission')
    ax1.set_ylabel('Action sensitivity (Q std across plausible actions)', 
                   color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # secondary: patient count — fades as n drops
    ax2 = ax1.twinx()
    ax2.fill_between(hours, n_patients, alpha=0.15, color='gray')
    ax2.set_ylabel('N patients at this hour', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, n_patients.max() * 4)  # compress so it doesn't dominate

    # mark where n drops below a meaningful threshold
    sparse_threshold = 200
    sparse_start = hours[n_patients < sparse_threshold]
    if len(sparse_start) > 0:
        ax1.axvline(sparse_start[0], color='gray', ls=':', lw=1.5,
                    label=f'N < {sparse_threshold} patients')

    # Add phase annotations
    ax1.axvspan(0, 250, alpha=0.05, color='blue', label='Phase 1: declining leverage')
    ax1.axvspan(330, 750, alpha=0.05, color='orange', label='Phase 2: rising leverage')
    ax1.axvspan(750, hours.max(), alpha=0.1, color='gray', label='Unreliable (low N)')

    ax1.annotate('Early\nresuscitation', xy=(5, mean_sens[0]), 
                fontsize=9, color='blue')
    ax1.annotate('Late ICU\nphase?', xy=(400, mean_sens[hours >= 330][0]),
                fontsize=9, color='orange')

    ax1.set_title('When does the treatment decision carry most leverage?')
    ax1.legend(loc='upper right')
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "temporal_plot.png"), 
                    dpi=150, bbox_inches='tight')
    return fig, ax1

def plot_direction_of_improvement(direction_per_dim, headroom_per_dim, 
                                   trust_mask, temporal_index, actions,
                                   calibrated_threshold,
                                   early_cutoff=6, save_dir=None):
    """
    direction_per_dim: (N, action_dim) signed direction of best alternative
    headroom_per_dim:  (N, action_dim) per-dim headroom
    trust_mask:        (N,) bool — low disagreement states
    temporal_index:    (N,) hour within trajectory
    actions:           (N, action_dim) clinician actions
    early_cutoff:      hours threshold for early vs late phase
    """
    dim_names = ['Fluid', 'Vasopressor']
    early = temporal_index < early_cutoff
    late  = temporal_index >= early_cutoff

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for d, name in enumerate(dim_names):
        hr_d  = headroom_per_dim[:, d]
        dir_d = direction_per_dim[:, d]
        clin_d = actions[:, d]

        # meaningful = genuine headroom + trusted
        # meaningful = (hr_d > np.quantile(hr_d[hr_d > 0], 0.5)) & trust_mask
        meaningful = (hr_d  > calibrated_threshold) & trust_mask

        # --- Row 0: overall direction histogram with early/late overlay ---
        ax = axes[0, d]
        # overall in gray
        ax.hist(dir_d[meaningful], bins=60, color='gray', alpha=0.4,
                label='All hours', density=True)
        # early phase
        m_early = meaningful & early
        m_late  = meaningful & late
        if m_early.sum() > 20:
            ax.hist(dir_d[m_early], bins=60, alpha=0.5, density=True,
                    label=f'Early (0–{early_cutoff}h)', color='tab:blue')
        if m_late.sum() > 20:
            ax.hist(dir_d[m_late], bins=60, alpha=0.5, density=True,
                    label=f'Late (>{early_cutoff}h)', color='tab:orange')
        ax.axvline(0, color='k', ls='--', lw=1)
        ax.set_xlabel(f'Suggested Δ{name} (alt − clinician)')
        ax.set_ylabel('Density')
        pct_more_all   = (dir_d[meaningful] > 0).mean() * 100
        pct_more_early = (dir_d[m_early] > 0).mean() * 100 if m_early.sum() > 0 else float('nan')
        pct_more_late  = (dir_d[m_late]  > 0).mean() * 100 if m_late.sum()  > 0 else float('nan')
        ax.set_title(
            f'{name}: overall {pct_more_all:.0f}% more / {100-pct_more_all:.0f}% less\n'
            f'Early: {pct_more_early:.0f}% more  |  Late: {pct_more_late:.0f}% more'
        )
        ax.legend(fontsize=8)

        # --- Row 1: headroom and clinician action split by direction ---
        ax2 = axes[1, d]
        more_mask = meaningful & (dir_d > 0)
        less_mask = meaningful & (dir_d < 0)

        categories = ['Suggest more', 'Suggest less']
        hr_means   = [hr_d[more_mask].mean()   if more_mask.sum() > 0 else 0,
                      hr_d[less_mask].mean()   if less_mask.sum() > 0 else 0]
        hr_sems    = [hr_d[more_mask].std() / np.sqrt(more_mask.sum()) if more_mask.sum() > 0 else 0,
                      hr_d[less_mask].std() / np.sqrt(less_mask.sum()) if less_mask.sum() > 0 else 0]
        clin_means = [clin_d[more_mask].mean() if more_mask.sum() > 0 else 0,
                      clin_d[less_mask].mean() if less_mask.sum() > 0 else 0]
        clin_sems  = [clin_d[more_mask].std() / np.sqrt(more_mask.sum()) if more_mask.sum() > 0 else 0,
                      clin_d[less_mask].std() / np.sqrt(less_mask.sum()) if less_mask.sum() > 0 else 0]

        x = np.arange(len(categories))
        width = 0.35

        # headroom bars
        bars1 = ax2.bar(x - width/2, hr_means, width, yerr=hr_sems,
                        capsize=4, label='Mean headroom', color='tab:blue', alpha=0.7)
        ax2.set_ylabel('Mean headroom', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # clinician action on secondary axis
        ax2b = ax2.twinx()
        bars2 = ax2b.bar(x + width/2, clin_means, width, yerr=clin_sems,
                         capsize=4, label=f'Mean clinician {name}',
                         color='tab:orange', alpha=0.7)
        ax2b.set_ylabel(f'Mean clinician {name}', color='tab:orange')
        ax2b.tick_params(axis='y', labelcolor='tab:orange')

        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_title(f'{name}: headroom and clinician dose by direction')

        # combined legend
        lines = [bars1, bars2]
        labels = [b.get_label() for b in lines]
        ax2.legend(lines, labels, loc='upper right', fontsize=8)

        # annotate n per group
        for xi, (mm, lm) in enumerate([(more_mask, less_mask)]):
            pass
        ax2.annotate(f'n={more_mask.sum():,}', xy=(x[0] - width/2, hr_means[0]),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='tab:blue')
        ax2.annotate(f'n={less_mask.sum():,}', xy=(x[1] - width/2, hr_means[1]),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=8, color='tab:blue')

    plt.suptitle('Direction of improvement: when and how much?', 
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "direction_of_improvement_plot.png"),
                    dpi=150, bbox_inches='tight')
    return fig, axes



def headroom_by_disagreement(headroom_joint, disagreement, disagree_thresh, 
                             calibrated_threshold, save_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    # scatter — subsample for visibility
    ax.scatter(disagreement[::20], headroom_joint[::20], s=4, alpha=0.2)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.axvline(disagree_thresh, color='r', ls=':', lw=1, label='Trust threshold')

    # binned mean headroom overlay
    n_bins   = 20
    bin_edges = np.quantile(disagreement, np.linspace(0, 1, n_bins + 1))
    bin_idx   = np.clip(np.digitize(disagreement, bin_edges[1:-1]), 0, n_bins - 1)
    bin_centres, mean_hr = [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 50:
            bin_centres.append(disagreement[m].mean())
            mean_hr.append(headroom_joint[m].mean())
    ax.plot(bin_centres, mean_hr, 'r-', lw=2, label='Mean headroom per bin')

    # correlation
    r,   p_r   = scipy_stats.pearsonr(disagreement,  headroom_joint)
    rho, p_rho = scipy_stats.spearmanr(disagreement, headroom_joint)

    # improvable at low vs high disagreement
    low_mask  = disagreement <  disagree_thresh
    high_mask = disagreement >= disagree_thresh
    pct_low  = (headroom_joint[low_mask]  > calibrated_threshold).mean() * 100
    pct_high = (headroom_joint[high_mask] > calibrated_threshold).mean() * 100

    # annotation block
    textstr = (
        f'Pearson r = {r:.3f} (p={p_r:.2e})\n'
        f'Spearman ρ = {rho:.3f} (p={p_rho:.2e})\n'
        f'% improvable (>{calibrated_threshold:.4f}):\n'
        f'  Low disagreement:  {pct_low:.2f}%\n'
        f'  High disagreement: {pct_high:.2f}%'
    )
    ax.text(0.98, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Q1–Q2 disagreement (uncertainty)')
    ax.set_ylabel('Headroom')
    ax.set_title('Is headroom concentrated where the model is uncertain?')
    ax.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "headroom_by_disagreement_plot.png"),
                    dpi=150, bbox_inches='tight')
    return fig, ax


def stratify_disagreement_by_mortality(disagreement, headroom_joint, 
                                        mortality_values, temporal_index,
                                        disagree_thresh, calibrated_threshold,
                                        save_dir=None):
    
    survivor_mask    = mortality_values == 0
    nonsurvivor_mask = mortality_values == 1
    low_mask         = disagreement < disagree_thresh
    high_mask        = disagreement >= disagree_thresh

    # --- 1. Summary table ---
    print("=== Disagreement stratified by mortality ===\n")
    for label, mask in [('Survivors', survivor_mask), 
                         ('Non-survivors', nonsurvivor_mask)]:
        print(f"{label} (n={mask.sum():,})")
        print(f"  Mean disagreement:        {disagreement[mask].mean():.4f}")
        print(f"  Median disagreement:      {np.median(disagreement[mask]):.4f}")
        print(f"  % above trust threshold:  "
              f"{(disagreement[mask] >= disagree_thresh).mean()*100:.1f}%")
        print(f"  % improvable (low disagree, >{calibrated_threshold:.4f}):  "
              f"{(headroom_joint[mask & low_mask] > calibrated_threshold).mean()*100:.2f}%")
        print(f"  % improvable (high disagree, >{calibrated_threshold:.4f}): "
              f"{(headroom_joint[mask & high_mask] > calibrated_threshold).mean()*100:.2f}%")
        print(f"  Mean temporal index:      {temporal_index[mask].mean():.1f}h")
        print()

    # --- 2. Figure: 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: disagreement distribution by mortality
    axes[0].hist(disagreement[survivor_mask],    bins=80, alpha=0.6,
                 density=True, label='Survivors',     color='tab:blue')
    axes[0].hist(disagreement[nonsurvivor_mask], bins=80, alpha=0.6,
                 density=True, label='Non-survivors', color='tab:orange')
    axes[0].axvline(disagree_thresh, color='r', ls=':', lw=1.5,
                    label='Trust threshold')
    axes[0].set_xlabel('Q1–Q2 disagreement')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Disagreement distribution by outcome')
    axes[0].legend()

    # Panel 2: % above trust threshold and % improvable by mortality — bar chart
    categories = ['Survivors', 'Non-survivors']
    masks       = [survivor_mask, nonsurvivor_mask]

    pct_high_disagree = [
        (disagreement[m] >= disagree_thresh).mean() * 100 for m in masks
    ]
    pct_improvable_low = [
        (headroom_joint[m & low_mask] > calibrated_threshold).mean() * 100 for m in masks
    ]
    pct_improvable_high = [
        (headroom_joint[m & high_mask] > calibrated_threshold).mean() * 100 for m in masks
    ]

    x     = np.arange(len(categories))
    width = 0.25

    axes[1].bar(x - width,     pct_high_disagree,   width, label='% high disagreement',
                color='tab:red',    alpha=0.7)
    axes[1].bar(x,             pct_improvable_low,  width, label='% improvable (low disagree)',
                color='tab:blue',   alpha=0.7)
    axes[1].bar(x + width,     pct_improvable_high, width, label='% improvable (high disagree)',
                color='tab:orange', alpha=0.7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel('%')
    axes[1].set_title('Uncertainty and improvability by outcome')
    axes[1].legend(fontsize=8)

    # annotate values on bars
    for offset, vals in zip([-width, 0, width],
                             [pct_high_disagree, pct_improvable_low, pct_improvable_high]):
        for xi, val in enumerate(vals):
            axes[1].annotate(f'{val:.1f}%',
                            xy=(xi + offset, val),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', fontsize=7)

    # Panel 3: disagreement vs temporal index by mortality
    # show how disagreement evolves over the trajectory for each group
    max_hours = min(int(temporal_index.max()) + 1, 250)
    min_n     = 20

    for mask, label, color in [(survivor_mask,    'Survivors',     'tab:blue'),
                                (nonsurvivor_mask, 'Non-survivors', 'tab:orange')]:
        hours_plot, mean_d, sem_d = [], [], []
        for t in range(max_hours):
            m = (temporal_index == t) & mask
            n = m.sum()
            if n >= min_n:
                hours_plot.append(t)
                mean_d.append(disagreement[m].mean())
                sem_d.append(disagreement[m].std() / np.sqrt(n))
        hours_plot = np.array(hours_plot)
        mean_d     = np.array(mean_d)
        sem_d      = np.array(sem_d)
        axes[2].plot(hours_plot, mean_d, lw=2, label=label, color=color)
        axes[2].fill_between(hours_plot, mean_d - sem_d, mean_d + sem_d,
                             alpha=0.2, color=color)

    axes[2].axhline(disagree_thresh, color='r', ls=':', lw=1.5,
                    label='Trust threshold')
    axes[2].set_xlabel('Hours since ICU admission')
    axes[2].set_ylabel('Mean Q1–Q2 disagreement')
    axes[2].set_title('How does model uncertainty evolve over time?')
    axes[2].legend(fontsize=8)

    plt.suptitle('Critic uncertainty stratified by patient outcome', 
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "disagreement_by_mortality.png"),
                    dpi=150, bbox_inches='tight')
    return fig, axes



def plot_noise_floor_comparison(real_headroom, noise_headroom, reward_name="headroom",
                                  min_gap=0.01, stable_window=20, 
                                  threshold=None, thresholds=None, 
                                  gap=None, save_dir=None):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: CCDF comparison with calibrated threshold marked ---
    ax = axes[0]
    for d, label, color in [
        (real_headroom,  'Clinician-centred headroom', 'tab:blue'),
        (noise_headroom, 'Noise floor (random baseline)', 'tab:red'),
    ]:
        d_sorted = np.sort(d)
        ccdf = 1.0 - np.arange(len(d_sorted)) / len(d_sorted)
        ax.plot(d_sorted, ccdf, lw=2, label=label, color=color)

    ax.set_yscale('log')
    ax.set_xlabel('Headroom threshold')
    ax.set_ylabel('Fraction of states with headroom > threshold')
    ax.set_title(f'Noise floor calibration — {reward_name}')

    if threshold is not None:
        ax.axvline(threshold, color='k', ls=':', lw=1.5,
                   label=f'Calibrated threshold = {threshold:.4f}')
        pct_above = (real_headroom > threshold).mean() * 100
        ax.annotate(f'{pct_above:.2f}% of states\nabove threshold',
                    xy=(threshold, (real_headroom > threshold).mean()),
                    xytext=(15, 15), textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='->', lw=0.8))

    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    # --- Right: the gap itself, showing where it stabilises ---
    ax2 = axes[1]
    ax2.plot(thresholds, gap, lw=2, color='tab:purple')
    ax2.axhline(min_gap, color='gray', ls='--', lw=1, label=f'Min gap = {min_gap}')
    if threshold is not None:
        ax2.axvline(threshold, color='k', ls=':', lw=1.5,
                    label=f'Calibrated threshold = {threshold:.4f}')
    ax2.set_xlabel('Headroom threshold')
    ax2.set_ylabel('CCDF gap (real − noise)')
    ax2.set_title('Divergence between real and noise-floor CCDFs')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"noise_floor_{reward_name}.png"),
                    dpi=150, bbox_inches='tight')

    return fig, axes, threshold


def plot_figures(action_analysis, save_dir=None):
    threshold = _load_npz_scalar(action_analysis, 'threshold')

    # ---- guards: only interpret headroom/direction where the model is trustworthy ----
    disagree_thresh = np.quantile(action_analysis['disagreement'], 0.75)   # flag top-quartile uncertainty
    trust_mask = action_analysis['disagreement'] < disagree_thresh

    survivor   = action_analysis['mortality_values'] == 0
    nonsurvivor = action_analysis['mortality_values'] == 1

    # plot calibration
    plot_calibration(
        action_analysis['q_values'], 
        action_analysis['mortality_values'], 
        survivor,
        nonsurvivor,
        save_dir=save_dir
    )

    # plot headroom 
    plot_headroom(
        action_analysis['headroom_joint'], 
        trust_mask, 
        survivor, 
        nonsurvivor,
        threshold,
        save_dir=save_dir
    )

    # plot temporal structure
    plot_temporal_structure(
        action_analysis['sensitivity_joint'], 
        action_analysis['temporal_index'], 
        trust_mask, 
        save_dir=save_dir
    )

    # plot direction of improvement
    plot_direction_of_improvement(
        action_analysis['direction_per_dim'], 
        action_analysis['headroom_per_dim'], 
        trust_mask, 
        action_analysis['temporal_index'],
        actions=action_analysis['clinician_actions'],
        calibrated_threshold=threshold,
        save_dir=save_dir
    )

    # plot headroom by disagreement
    headroom_by_disagreement(
        action_analysis['headroom_joint'], 
        action_analysis['disagreement'], 
        disagree_thresh, 
        threshold,
        save_dir=save_dir
    )

    stratify_disagreement_by_mortality(
        action_analysis['disagreement'], 
        action_analysis['headroom_joint'], 
        action_analysis['mortality_values'], 
        action_analysis['temporal_index'],
        disagree_thresh, 
        threshold,
        save_dir=save_dir
    )

    plot_noise_floor_comparison(
        action_analysis['headroom_joint'],
        action_analysis['noise_headroom'],
        threshold=threshold,
        thresholds=action_analysis['thresholds'],
        gap=action_analysis['gap'],
        save_dir=save_dir
    )


    return


# =============================================================================
# Comparing SARSA critics seed for seed (e.g. early-stopped vs long-trained)
# =============================================================================
#
# Question: do better-trained judges change what we conclude? Three things are compared per seed on one split:
#   1. critic quality  - Bellman residual and level against observed discounted returns (interior rows only)
#   2. the advantage   - Q(s, pi(s)) - Q(s, a_clinician) of the final CQL policy under each critic
#   3. agreement       - row-wise correlation of Q and of the advantage between the two critics
# The pass criteria for (1) are arguments, so they are fixed before looking: |gap_interior| < gap_tol and
# |mean_td| < td_tol (a mean TD of 0.0025 is about 0.1 of level error at gamma = 0.975).


def _reward_scaling(critic) -> float:
    rs = critic.record.get("reward_scaling")
    return 1.0 if rs is None else float(rs)


def critic_td_summary(critic, processor, next_actions=None, huber_beta: float = 1.0) -> Dict[str, float]:
    """
    Bellman residual of a critic on `processor`, with the TRAINER's conventions (basic_q.Q_Learner):
      target = r * reward_scaling + discount * (1 - done * mortality) * Q(s', a')
    i.e. only death is terminal; a discharge is truncated and bootstraps from the next state. Q is raw (unscaled).
    next_actions: the a' used in the target (default: the logged next action, i.e. SARSA's own target; pass the
    target policy's actions at s' to test an FQE critic against its policy).
    Returns mean / median / rmse of the TD error (target - Q; ~0 on average for a critic at its fixed point)
    and the mean Huber (smooth-L1) loss of it, comparable in size to the logged training / validation critic loss.
    """
    gamma = critic.record.get("discount")
    if gamma is None:
        raise KeyError(f"{critic.record.label}: no 'discount' in the run's hyperparameters")
    mort = D.mortality(processor)
    if mort is None:
        raise ValueError("processor has no mortality; needed for the trainer's done rule")
    s, a, s2 = D.states(processor), D.actions(processor), D.next_states(processor)
    a2 = D.next_actions(processor) if next_actions is None else next_actions
    r, done = D.rewards(processor) * _reward_scaling(critic), D.dones(processor)
    not_done = 1.0 - (done & mort)
    td = r + float(gamma) * not_done * critic.q(s2, a2) - critic.q(s, a)
    ad = np.abs(td)
    huber = np.where(ad < huber_beta, 0.5 * td ** 2 / huber_beta, ad - 0.5 * huber_beta)
    return {"mean_td": float(td.mean()), "median_td": float(np.median(td)), "rmse_td": float(np.sqrt((td ** 2).mean())),
            "huber_td": float(huber.mean())}


def compare_sarsa_critics(old_models, new_models, store, split="val", cql_models=None, n_boot=500, boot_seed=0,
                          tail_weight=0.05, batch_size=4096, verbose=True):
    """
    old_models, new_models : SARSA critics, {seed: LoadedModel} or one group (e.g. load_study_models(study)[()])
    store                  : data.DataStore; each seed's data comes from its own critic's data identity
    cql_models             : OPTIONAL final CQL models, {seed: LoadedModel}. Without them only critic quality is
                             compared; with them the advantage of the CQL policy under each critic is added.
    Returns (per_critic, per_seed):
      per_critic : one row per (seed, critic in {old, new}): mean / median / rmse / Huber of the TD error; interior level gap
                   (Q - observed return on rows with enough later rows, see mc_calibration_one_seed) with CI; corr_interior;
                   [with cql_models: normalised advantage with a stay-bootstrap CI and the share of rows with advantage > 0]
      per_seed   : agreement between the critics: corr of Q;
                   [with cql_models: corr and sign agreement of the per-row advantage, and the change in normalised
                   advantage (new - old) with a paired stay-bootstrap CI]
    """
    old, new = _by_seed(old_models), _by_seed(new_models)
    cql = _by_seed(cql_models) if cql_models is not None else None
    seeds = sorted(set(old) & set(new) & (set(cql) if cql is not None else set(old)), key=str)
    dropped = (set(old) | set(new) | (set(cql) if cql is not None else set())) - set(seeds)
    if not seeds:
        raise KeyError(f"no seed present in all model sets: old {sorted(old, key=str)}, new {sorted(new, key=str)}")
    if dropped:
        warnings.warn(f"seeds {sorted(dropped, key=str)} are not in all model sets and were skipped")

    per_critic, per_seed = [], []
    for seed in seeds:
        crit = {"old": old[seed], "new": new[seed]}
        ref = cql[seed] if cql is not None else crit["old"]          # defines the data for this seed
        for c in crit.values():
            check_judge_matches(ref, c)
            if _reward_scaling(c) != 1.0:
                warnings.warn(f"{c.record.label}: reward_scaling = {_reward_scaling(c)}; observed returns are not rescaled, "
                              f"so the level gap is not comparable")
        p = store.processor(ref, split)
        s, a_clin = D.states(p), D.actions(p)
        traj = D.trajectory_ids(p)
        inv = np.unique(traj, return_inverse=True)[1]
        J = int(inv.max()) + 1
        n = np.bincount(inv, minlength=J).astype(np.float64)
        idx = np.random.default_rng(boot_seed).integers(0, J, size=(n_boot, J), dtype=np.int32)
        den = n[idx].sum(1)
        a_pi = actor_actions(cql[seed], s, batch_size=batch_size) if cql is not None else None

        keep = {}
        for name, c in crit.items():
            td = critic_td_summary(c, p)
            mc = mc_calibration_one_seed(c, p, n_boot=n_boot, boot_seed=boot_seed, tail_weight=tail_weight,
                                         batch_size=batch_size)["row"]
            q_cl = c.q(s, a_clin, batch_size=batch_size, scaled=True)
            keep[name] = {"q": q_cl}
            row = {"seed": seed, "critic": name, **td,
                   "gap_interior": mc["gap_interior"], "gap_interior_lo": mc["gap_interior_lo"], "gap_interior_hi": mc["gap_interior_hi"],
                   "corr_interior": mc["corr_interior"], "Q_interior_mean": mc["Q_interior_mean"], "G_interior_mean": mc["G_interior_mean"],
                   "H_interior": mc["H_interior"]}
            if a_pi is not None:
                adv = c.q(s, a_pi, batch_size=batch_size, scaled=True) - q_cl
                iqr = max(float(np.quantile(q_cl, 0.75) - np.quantile(q_cl, 0.25)), 1e-6)
                boot = (np.bincount(inv, weights=adv, minlength=J)[idx].sum(1) / den) / iqr
                keep[name].update({"adv": adv, "boot": boot, "norm": float(adv.mean() / iqr)})
                row.update({"adv_all_norm": keep[name]["norm"], "frac_adv_pos": float((adv > 0).mean())})
                row["adv_all_norm_lo"], row["adv_all_norm_hi"] = _pct_ci(boot)
            per_critic.append(row)

        o, nw = keep["old"], keep["new"]
        pair = {"seed": seed, "corr_q": float(np.corrcoef(o["q"], nw["q"])[0, 1])}
        if a_pi is not None:
            pair.update({"corr_adv": float(np.corrcoef(o["adv"], nw["adv"])[0, 1]),
                         "sign_agree": float((np.sign(o["adv"]) == np.sign(nw["adv"])).mean()),
                         "adv_norm_old": o["norm"], "adv_norm_new": nw["norm"], "d_adv_norm": nw["norm"] - o["norm"]})
            pair["d_adv_norm_lo"], pair["d_adv_norm_hi"] = _pct_ci(nw["boot"] - o["boot"])
        per_seed.append(pair)
        if verbose:
            r = {x["critic"]: x for x in per_critic if x["seed"] == seed}
            msg = (f"  seed {seed}: gap old {r['old']['gap_interior']:+.3f} -> new {r['new']['gap_interior']:+.3f}   "
                   f"mean TD {r['old']['mean_td']:+.4f} -> {r['new']['mean_td']:+.4f}   corr(Q) {pair['corr_q']:.2f}")
            if a_pi is not None:
                msg += f"   adv_norm {pair['adv_norm_old']:.3f} -> {pair['adv_norm_new']:.3f}   corr(adv) {pair['corr_adv']:.2f}"
            print(msg)
    return pd.DataFrame(per_critic), pd.DataFrame(per_seed)


def plot_sarsa_comparison(per_critic: pd.DataFrame, per_seed: pd.DataFrame, gap_tol: float = 0.1, td_tol: float = 0.003,
                          title: str = "", save_path=None):
    """
    Old vs new critic per seed: level gap (with its pass band), mean TD (with its band), agreement, and, when the
    comparison was run with cql_models, the normalised advantage.
    """
    has_adv = "adv_all_norm" in per_critic
    n_panels = 4 if has_adv else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), constrained_layout=True)
    seeds = list(dict.fromkeys(per_critic.seed))
    x = np.arange(len(seeds))
    style = {"old": ("o", "C1", -0.12), "new": ("s", "C0", 0.12)}

    def col(name, c):
        d = per_critic[per_critic.critic == name].set_index("seed")
        return d.loc[seeds, c].to_numpy()

    ax = axes[0]
    ax.axhspan(-gap_tol, gap_tol, color="green", alpha=0.08); ax.axhline(0, color="grey", lw=1)
    for name, (mk, cl, off) in style.items():
        y = col(name, "gap_interior")
        ax.errorbar(x + off, y, yerr=[y - col(name, "gap_interior_lo"), col(name, "gap_interior_hi") - y], fmt=mk, color=cl, capsize=3, label=name)
    ax.set_title("level: Q - observed return (interior rows)"); ax.set_ylabel("health-score points"); ax.legend(frameon=False)

    ax = axes[1]
    ax.axhspan(-td_tol, td_tol, color="green", alpha=0.08); ax.axhline(0, color="grey", lw=1)
    for name, (mk, cl, off) in style.items():
        ax.plot(x + off, col(name, "mean_td"), mk, color=cl, label=name)
    ax.set_title("mean Bellman (TD) error"); ax.set_ylabel("reward units per step")

    k = 2
    if has_adv:
        ax = axes[k]; k += 1
        ax.axhline(0, color="grey", lw=1)
        for name, (mk, cl, off) in style.items():
            y = col(name, "adv_all_norm")
            ax.errorbar(x + off, y, yerr=[y - col(name, "adv_all_norm_lo"), col(name, "adv_all_norm_hi") - y], fmt=mk, color=cl, capsize=3, label=name)
        ax.set_title("advantage of the CQL policy"); ax.set_ylabel("advantage / IQR(Q at clinician actions)")

    ax = axes[k]
    ps = per_seed.set_index("seed").loc[seeds]
    ax.plot(x, ps.corr_q, "^", color="grey", label="corr of Q")
    if has_adv:
        ax.plot(x - 0.06, ps.corr_adv, "o", label="corr of per-row advantage")
        ax.plot(x + 0.06, ps.sign_agree, "s", label="sign agreement")
    ax.set_ylim(0, 1.02); ax.set_title("agreement old vs new"); ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xticks(x, [str(kk) for kk in seeds]); ax.set_xlabel("seed")
    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    if args.experiment_variant == "hyperparameter_search_evaluation":
        hyperparameter_search_evaluation(args)
    
    elif args.experiment_variant == "evaluate_final_model":
        action_analysis, save_dir = evaluate_final_model(args)
        plot_figures(action_analysis, save_dir=save_dir)


if __name__ == "__main__":
    main()