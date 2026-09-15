import argparse
import json
import os
import pickle
from pathlib import Path
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd

import torch
import numpy as np
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .models.td3_bc import Actor as TD3Actor
from .models.conservative_q_learning import Actor as CQLActor
from .models.conservative_q_learning import Critic as CQLCritic
from .models.cql_with_logprobs import Actor as CQL_with_logprobs_Actor
from .models.iql import Actor as IQLActor
from .models.iql import Critic as IQLCritic
from .models.iql import ValueNetwork as IQLValue
from .models.bc_with_logprobs_v2 import Actor as BCActor
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from ..setup_environment import load_paths


rl_models_path = "/home/jake/Documents/AI_Clinician/rl_models/" 


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


def load_actor(config_manager, model_path=None, load_critic=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                        f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions, "final_model")
    
        # get folder inside model_path
        model_folder = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))][0]
        model_path = os.path.join(model_path, model_folder)

    # configuration file is in the parent directory of the model_path
    configuration_path = os.path.join(os.path.dirname(model_path), "config_manager.pkl")
    actor_configuration = load_configuration(configuration_path)

    final_model_path = os.path.join(model_path, "final_model.pt")

    # load results.json from model_folder, which has state_dim 
    results_path = os.path.join(model_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            hyperparams = results["hyperparameters"]
            # hidden_dim = hyperparams["hidden_dim"]

    if actor_configuration.model_type == "td3_bc":
        actor = TD3Actor(512, 2).to(device)
    elif actor_configuration.model_type == "conservative_q_learning":
        actor = CQLActor(512, hyperparams["hidden_dim"]).to(device)
    elif actor_configuration.model_type == "iql":
        actor = IQLActor(512, hyperparams["hidden_dim"]).to(device)
    elif actor_configuration.model_type == "bc_with_logprobs":
        actor = BCActor(512).to(device)
    elif actor_configuration.model_type == "cql_with_logprobs":
        actor = CQL_with_logprobs_Actor(512, hidden_dim=hyperparams["hidden_dim"]).to(device)
    else:
        raise ValueError(f"Unknown model type: {actor_configuration.model_type}")

    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # load critic as well
    if load_critic:
        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.eval()
    else:
        critic = None

    return actor, critic, actor_configuration, hyperparams, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL models on SARSA CHOC")

    # Keyword Arguments (Match your launch.json)
    parser.add_argument("--hospital", type=str, default="choc")
    parser.add_argument("--rl_type", type=str, choices=["td3_bc", 
        "conservative_q_learning", "bc", "implicit_q_learning", "cql_with_log_probs"], required=True)
    parser.add_argument("--experiment_type", type=str, default="model_evaluation")
    parser.add_argument("--model_type", type=str, default="basic_q")
    parser.add_argument("--mode", type=str, default="single_split")
    parser.add_argument("--split_num", type=int, default=0)
    parser.add_argument("--cohort", type=str, default="sepsis")
    parser.add_argument("--actions", type=str, default="vis-intake")
    parser.add_argument("--reward_type", type=str, default="health_score_change_only")
    parser.add_argument("--discount", type=float, default=0.975)
    
    # The Flag (Action: store_true means if it's in the args, it becomes True)
    parser.add_argument("--testing", action="store_true", help="Run in test mode")

    return parser.parse_args()


def generate_actions(rl_actor, states, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
    
    loader = DataLoader(states, batch_size=1024, shuffle=False)

    all_rl_actions = []
    with torch.no_grad():        
        for batch in loader:
            batch_states = batch.to(device)
            rl_actions = rl_actor.act(batch_states, device=device)
            all_rl_actions.append(rl_actions)

    all_rl_actions = np.concatenate(all_rl_actions, axis=0)
    # convert back to torch tensor
    all_rl_actions = torch.from_numpy(all_rl_actions).float().to(device)
    return all_rl_actions


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


def generate_action_logprobs(
        test_processor, 
        batch_size=1024, 
        actor=None,
        bc_model=None,
        device="cuda" if torch.cuda.is_available() else "cpu"       
):
    # loop through trajectories in test_processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(test_processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = test_processor.dynamic_deephit_dataset["stay_ids"]

    # create storage
    logprobs = torch.zeros(test_processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)

    for stay_id in stay_ids:
        states, traj_actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=True) 
               
        n_batches = int(np.ceil(states.shape[0] / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, states.shape[0])
            states_batch = states[start_idx:end_idx]
            actions_batch = traj_actions[start_idx:end_idx]

            if bc_model is not None:
                logprob = actor.log_prob_for_wis(states_batch, actions_batch, bc_model)
            else:
                logprob = actor.log_prob_of_action(states_batch, actions_batch)
            logprobs[indices[start_idx:end_idx]] = logprob
                   
    return logprobs


def calculate_wis(test_processor, logprobs_rl, logprobs_bc, config_manager):
    # loop through trajectories in test_processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(test_processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = test_processor.dynamic_deephit_dataset["stay_ids"]

    # per-timestep, for all transitions in the dataset
    log_rho = logprobs_rl - logprobs_bc                                     # shape (N,)

    unclipped_w_tau = []
    w_tau = []
    G_tau = []
    for stay_id in stay_ids:
        states, traj_actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=False) 

        log_rho_traj = log_rho[indices]  # shape (T,)
        log_w_tau = np.sum(log_rho_traj)  # sum in log space
        clipped_log_w_tau = np.clip(log_w_tau, a_min=-20, a_max=20)  # clip to avoid overflow
        w_tau.append(np.exp(clipped_log_w_tau))  # exponentiate ONCE, at the end
        unclipped_w_tau.append(np.exp(log_w_tau))
        G_tau.append(np.sum([config_manager.discount**t * rewards[t] for t in range(len(rewards))]))

    
    wis_results = bootstrap_wis_vs_behavior(w_tau, G_tau, n_bootstrap=2000, seed=0)

    # V_wis = np.sum([w * G for w, G in zip(w_tau, G_tau)]) / np.sum(w_tau)
    # V_behaviour = np.mean(G_tau)    
    # print(f"V_wis: {V_wis:.4f}, V_behaviour: {V_behaviour:.4f}")
    # ess = (np.array(w_tau).sum() ** 2) / (np.array(w_tau) ** 2).sum()

    # calculate the mean and std of unclipped_w_tau
    unclipped_w_tau_mean = np.mean(unclipped_w_tau)
    unclipped_w_tau_std = np.std(unclipped_w_tau) 
    unclipped_w_tau_median = np.median(unclipped_w_tau)
    unclipped_w_tau_min = np.min(unclipped_w_tau)
    unclipped_w_tau_max = np.max(unclipped_w_tau)  
    print(f"Unclipped w_tau mean: {unclipped_w_tau_mean:.4f}, std: {unclipped_w_tau_std:.4f}")
    print(f"Unclipped w_tau median: {unclipped_w_tau_median:.4f}, min: {unclipped_w_tau_min:.4f}, max: {unclipped_w_tau_max:.4f}")

    # return V_wis, V_behaviour, ess
    return wis_results


def bootstrap_wis_vs_behavior(w_tau, G_tau, n_bootstrap=2000, seed=0):
    """
    Paired bootstrap over trajectories to test whether V_wis differs
    significantly from V_behavior (plain average of G_tau).
    w_tau, G_tau: same-length lists/arrays, one entry per trajectory,
    already computed as in calculate_wis().
    """
    rng = np.random.default_rng(seed)
    w_tau = np.array(w_tau)
    G_tau = np.array(G_tau)
    n = len(G_tau)

    diffs = np.zeros(n_bootstrap)
    wis_samples = np.zeros(n_bootstrap)
    behavior_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # resample trajectory indices with replacement
        w_b = w_tau[idx]
        G_b = G_tau[idx]

        v_wis_b = np.sum(w_b * G_b) / np.sum(w_b)
        v_behavior_b = np.mean(G_b)

        wis_samples[b] = v_wis_b
        behavior_samples[b] = v_behavior_b
        diffs[b] = v_wis_b - v_behavior_b

    # point estimates (from the actual, unresampled data)
    v_wis_point = np.sum(w_tau * G_tau) / np.sum(w_tau)
    v_behavior_point = np.mean(G_tau)
    diff_point = v_wis_point - v_behavior_point

    # 95% percentile CI on the difference
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    # two-sided bootstrap p-value: proportion of bootstrap diffs on the 
    # opposite side of 0 from the point estimate, doubled
    if diff_point >= 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)
    p_value = min(p_value, 1.0)

    return {
        "v_wis": v_wis_point,
        "v_behavior": v_behavior_point,
        "diff": diff_point,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "p_value": p_value,
        "significant_at_05": not (ci_low <= 0 <= ci_high),
    }

def plot_action_distributions(actions, model_path):
    # 2 subplots: one for each action dimension
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fluids = actions[:, 0].numpy()
    # remove zeros
    fluids_nonzero = fluids[fluids != 0]
    print(f"min fluids: {np.min(fluids_nonzero)}, max fluids: {np.max(fluids_nonzero)}")

    # plot histogram of non-zero fluids
    axs[0].hist(fluids_nonzero, bins=1000, alpha=0.7, color='blue', edgecolor='black')
    axs[0].set_title('Action Dimension 1 Distribution (Non-zero Fluids)')
    axs[0].set_xlabel('Action Value')
    axs[0].set_ylabel('Frequency')

    
    vasopressors = actions[:, 1].numpy()
    # remove zeros
    vasopressors_nonzero = vasopressors[vasopressors != 0]
    print(f"min vasopressors: {np.min(vasopressors_nonzero)}, max vasopressors: {np.max(vasopressors_nonzero)}")

    # plot histogram of non-zero vasopressors
    axs[1].hist(vasopressors_nonzero, bins=1000, alpha=0.7, color='red', edgecolor='black')
    axs[1].set_title('Action Dimension 2 Distribution (Non-zero Vasopressors)')
    axs[1].set_xlabel('Action Value')
    axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(model_path, "action_distributions.png"))
    plt.close()




def wis_multiple_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    # create configuration
    config_manager = ConfigManager(args)
   
    # get the path to the model_type model (i.e. bc_with_log_probs)
    bc_model_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.model_type,
                    f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions,
                    config_manager.reward_type, f"discount_{config_manager.discount}", "final_model")
    # get the folder inside bc_model_path
    bc_folders = [f for f in os.listdir(bc_model_path) if os.path.isdir(os.path.join(bc_model_path, f))]
    if len(bc_folders) != 1:
        raise ValueError(f"Expected exactly one folder in {bc_model_path}, but found {len(bc_folders)}: {bc_folders}")
    bc_model_path = os.path.join(bc_model_path, bc_folders[0])
    # load results.json from bc_model_folder
    with open(os.path.join(bc_model_path, "results.json"), "r") as f:
        bc_results = json.load(f)
    hyperparams = bc_results["hyperparameters"]
       
    # load the data
    train_processor, _, test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, hyperparams)
    test_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
    
    # create rl dataset
    batch_size = 1024
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams)

    # plot action distributions
    plot_action_distributions(train_processor.rl_dataset['actions'], bc_model_path)
    
    
    
    # load the BC model
    bc_model, _, bc_configuration, bc_hyperparams, bc_model_folder = load_actor(
        config_manager, model_path=bc_model_path, load_critic=False
    )

    # generate the log probabilities of the clinician actions under the BC model
    logprobs_bc = generate_action_logprobs(
        test_processor, 
        batch_size=batch_size, 
        actor=bc_model
    ).cpu().detach().numpy()
        
    
    # get the RL model directories
    rl_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                    f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions,
                    config_manager.reward_type, f"discount_{config_manager.discount}")
    rl_folders = [f for f in os.listdir(rl_path) if os.path.isdir(os.path.join(rl_path, f))]
    if "final_model" in rl_folders or "*_run" in rl_folders:
        rl_folders.remove("final_model")
        rl_folders = [f for f in rl_folders if not f.endswith("_run")]

    # sarsa_advantage_results =[]
    # rl_advantage_results = []

    for rl_folder in rl_folders:       
        model_path = os.path.join(rl_path, rl_folder)
        
        # load the configuration
        configuration_path = os.path.join(model_path, "config_manager.pkl")
        model_cfg = load_configuration(configuration_path)

        # find directories within model_path that begin with "local" or "unknown"
        model_subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and (f.startswith("local") or f.startswith("unknown"))]

        for subfolder in model_subfolders:
            model_subfolder_path = os.path.join(model_path, subfolder)
            
            print(f"Evaluating model in folder: {model_subfolder_path}")
            rl_model, _, rl_configuration, rl_hyperparams, rl_model_folder = load_actor(
                config_manager, model_path=model_subfolder_path, load_critic=False
            )

            # generate the log probabilities of the clinician actions under the BC model
            logprobs_rl = generate_action_logprobs(
                test_processor, 
                batch_size=batch_size, 
                actor=rl_model,
                bc_model=bc_model
            ).cpu().detach().numpy()

            calculate_wis(test_processor, logprobs_rl, logprobs_bc, config_manager)







def main():
    
    wis_multiple_models()  # Set to None to evaluate all models, or specify a model directory name


if __name__ == "__main__":
    main()