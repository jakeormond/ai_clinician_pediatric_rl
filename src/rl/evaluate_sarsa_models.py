import argparse
import json
import os
from typing_extensions import final

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score
from pathlib import Path
import pickle
from .training import create_buffer
from .models.shared_components import TrainerUtils
from ..setup_environment import load_paths


RL_MODELS_PATH = "/home/jake/Documents/AI_Clinician/models/rl_models/choc" 


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
        return states, actions, rewards, next_states, next_actions, dones, mortalities

    # convert to torch tensors
    states = torch.tensor(states, dtype=torch.float32).to('cuda')
    actions = torch.tensor(actions, dtype=torch.float32).to('cuda')
    rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda')
    next_states = torch.tensor(next_states, dtype=torch.float32).to('cuda')
    next_actions = torch.tensor(next_actions, dtype=torch.float32).to('cuda')
    dones = torch.tensor(dones, dtype=torch.float32).to('cuda')
    mortalities = torch.tensor(mortalities, dtype=torch.float32).to('cuda')

    return states, actions, rewards, next_states, next_actions, dones, mortalities



@torch.no_grad()
def hyperparameter_search_evaluation(args=None):
    
    search_results = []
    
    models_dir = os.path.join(RL_MODELS_PATH, 'basic_q', 
        f"split_{args.split_num}", args.cohort, args.actions)

    # get the model directories
    folder_paths = []
    search_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    
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
    df.to_csv(os.path.join(models_dir, "hyperparameter_search_evaluation_results.csv"), index=False)    


def evaluate_final_model(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = os.path.join(RL_MODELS_PATH, 'basic_q', 
        f"split_{args.split_num}", args.cohort, args.actions, "final_model")
    # get the model dir inside the final_model folder
    inner_folders = [inner_f for inner_f in os.listdir(model_dir) 
                     if os.path.isdir(os.path.join(model_dir, inner_f))]
    if len(inner_folders) == 0:
        print(f"No model found in folder: {model_dir}")
        return
    elif len(inner_folders) > 1:
        print(f"Multiple models found in folder: {model_dir}, evaluating the first one: {inner_folders[0]}")
        return
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
    batch_size = 1024
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams)

    # load the model
    sarsa_model = load_sarsa_critic(test_processor, hyperparams, model_path=model_path)

    # get the actions and get their standard deviation
    actions = test_processor.rl_dataset["actions"]
    action_dim = actions.shape[1]
    
    action_std = torch.tensor(actions.std(axis=0), dtype=torch.float32, device=device)
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
    disagreement    = []      # |Q1 - Q2| at clinician action
    headroom_per_dim = []     # (N, action_dim): best single-dim alt - clinician
    direction_per_dim = []    # (N, action_dim): signed (best_alt_action - clinician_action) per dim
    temporal_index  = [] 
    
    for stay_id in stay_ids:

        states, actions, rewards, next_states, next_actions, dones, mortalities = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=True) 

        n_batches = int(np.ceil(states.shape[0] / batch_size))
        
        # storage for q_values, mortalities, and headroom for each batch
        q_for_stay        = []
        mort_for_stay     = mortalities
        hr_joint_stay     = []
        sens_joint_stay   = []
        disagree_stay     = []
        hr_dim_stay       = []
        dir_dim_stay      = []     # position within trajectory

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, states.shape[0])
            states_batch = states[start_idx:end_idx]
            actions_batch = actions[start_idx:end_idx]

            # evaluate the model on the batch
            with torch.no_grad():
                q1_clin, q2_clin = sarsa_model(states_batch, actions_batch)
                q_clin = (q1_clin + q2_clin) / 2.0
                disagree = (q1_clin - q2_clin).abs()

                # --- joint perturbation: both dims at once ---
                best_joint = torch.full_like(q_clin, float('-inf'))
                joint_q_samples = []
                for _ in range(K):
                    noise = torch.randn_like(actions_batch) * action_std
                    alt   = (actions_batch + noise).clamp(0.0, 1.0)
                    q1_a, q2_a = sarsa_model(states_batch, alt)
                    q_a = (q1_a + q2_a) / 2.0
                    best_joint = torch.maximum(best_joint, q_a)
                    joint_q_samples.append(q_a)
                
                joint_q_samples = torch.cat(joint_q_samples, dim=1)   # (B, K)
                hr_joint   = best_joint - q_clin
                sens_joint = joint_q_samples.std(dim=1, keepdim=True)  # (B,1)

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
        headroom_per_dim.append(torch.cat(hr_dim_stay).cpu().numpy())
        direction_per_dim.append(torch.cat(dir_dim_stay).cpu().numpy())
        temporal_index.append(np.arange(n_t))
    
    # concatenate across stays
    q_values          = np.concatenate(q_values).ravel()
    mortality_values  = np.concatenate(mortality_values).ravel()
    headroom_joint    = np.concatenate(headroom_joint).ravel()
    sensitivity_joint = np.concatenate(sensitivity_joint).ravel()
    disagreement      = np.concatenate(disagreement).ravel()
    headroom_per_dim  = np.concatenate(headroom_per_dim)    # (N, action_dim)
    direction_per_dim = np.concatenate(direction_per_dim)   # (N, action_dim)
    temporal_index    = np.concatenate(temporal_index).ravel()

    # calculate auc for q_values and mortality_values
    y_true = 1 - mortality_values  # Assuming mortality=1 means death, we want to predict survival
    y_score = q_values.ravel()
    auc = roc_auc_score(y_true, y_score)
    print(f"Final model AUC: {auc:.4f}")

    # calculate auprc for q_values and mortality_values
    precision, recall, _ = stats.precision_recall_curve(y_true, y_score)
    auprc = stats.auc(recall, precision)
    print(f"Final model AUPRC: {auprc:.4f}")


    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    if args.experiment_variant == "hyperparameter_search_evaluation":
        hyperparameter_search_evaluation(args)
    
    elif args.experiment_variant == "evaluate_final_model":
        evaluate_final_model(args)


if __name__ == "__main__":
    main()