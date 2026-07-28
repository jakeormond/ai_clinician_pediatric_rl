import argparse
import json
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5

import torch
import numpy as np
from ..config_manager import ConfigManager   
from .models.basic_q import Critic
from .models.td3_bc import Actor as TD3Actor
from .models.cql_with_logprobs import Actor as CQLActor
from .models.cql_with_logprobs import Critic as CQLCritic
from .models.iql import Actor as IQLActor
from .models.iql import Critic as IQLCritic
from .models.iql import ValueNetwork as IQLValue
from .training import create_processor_from_shards
from .replay_buffer import RLDataset
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from ..setup_environment import load_paths




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


def load_test_data(config_manager, hyperparams):
    # load the data
    train_processor, _, test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, hyperparams)
    test_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
    
    # create rl dataset
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams)

    return test_processor


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


def load_sarsa_critic(model_dir=None, reward_type='terminal'):
    if model_dir is None:
        if reward_type == 'terminal':
            sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/terminal_only/final_model_terminal_discount_0.998"
        elif reward_type == 'health_score_change':
            sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/health_score_change_only/final_model_discount_0.975"
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")

        inner_directories = [f for f in os.listdir(sarsa_dir) if os.path.isdir(os.path.join(sarsa_dir, f))]
        model_dir = os.path.join(sarsa_dir, inner_directories[0])

        # if inner_directories[0] begins with "local" or "unknown", then we stop
        if not (inner_directories[0].startswith("local") or inner_directories[0].startswith("unknown")):  

            inner_directories = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(sarsa_dir, f))]
            if len(inner_directories) != 1:
                raise ValueError(f"Expected exactly one inner directory in {model_dir}, but found {len(inner_directories)}: {inner_directories}")
    
            model_dir = os.path.join(model_dir, inner_directories[0])
        
    configuration_path = os.path.join(model_dir, "config_manager.pkl")
    # check if config_manager.pkl exists
    if not os.path.exists(configuration_path):
        # then it's in the parent directory
        configuration_path = os.path.join(os.path.dirname(model_dir), "config_manager.pkl")
    
    config_manager = load_configuration(configuration_path)

    # load results.json
    hyperparams_path = os.path.join(model_dir, "results.json")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"results.json not found in {sarsa_dir}")
    with open(hyperparams_path, "r") as f:
        results = json.load(f)
        hyperparams = results["hyperparameters"]   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_model_path = os.path.join(model_dir, "final_model.pt")
    hidden_dim = hyperparams["hidden_dim"]

    sarsa_critic = Critic(512,2, hidden_dim=hidden_dim).to(device)
    sarsa_critic.to(device)
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['critic_state_dict'] 
    sarsa_critic.load_state_dict(state_dict)
    sarsa_critic.eval()

    # load hyperparameters from results.json
    results_path = os.path.join(model_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)  
    sarsa_hyperparams = results["hyperparameters"]
        
    return sarsa_critic, config_manager, sarsa_hyperparams


def load_actor_and_critic(model_dir=None, model_type='cql'):
    if model_dir is None:
        terminal_cql_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/cql_with_log_probs/split_0/sepsis/intake-vis/terminal_only/final_model_terminal_discount_0.998"
        inner_directories = [f for f in os.listdir(terminal_cql_dir) if os.path.isdir(os.path.join(terminal_cql_dir, f))]
        terminal_cql_dir = os.path.join(terminal_cql_dir, inner_directories[0])
        configuration_path = os.path.join(terminal_cql_dir, "config_manager.pkl")
        config_manager = load_configuration(configuration_path)
    
        inner_directories = [f for f in os.listdir(terminal_cql_dir) if os.path.isdir(os.path.join(terminal_cql_dir, f))]
        if len(inner_directories) != 1:
            raise ValueError(f"Expected exactly one inner directory in {terminal_cql_dir}, but found {len(inner_directories)}: {inner_directories}")
    
        model_dir = os.path.join(terminal_cql_dir, inner_directories[0])

    else: 
        # load config_manager.pkl from parent dir
        configuration_path = os.path.join(os.path.dirname(model_dir), "config_manager.pkl")
        config_manager = load_configuration(configuration_path)

    # load results.json
    hyperparams_path = os.path.join(model_dir, "results.json")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"results.json not found in {terminal_cql_dir}")
    with open(hyperparams_path, "r") as f:
        results = json.load(f)
        hyperparams = results["hyperparameters"]   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_model_path = os.path.join(model_dir, "final_model.pt")
    
    if model_type == "cql":
        actor = CQLActor(512, hyperparams["hidden_dim"]).to(device)
        critic = CQLCritic(512, 2, hidden_dim=hyperparams["hidden_dim"]).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types are 'cql' and 'iql'.")

    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # load critic as well
    critic.load_state_dict(checkpoint['critic_state_dict'])
    critic.eval()

    return actor, critic, config_manager, hyperparams


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



def generate_model_outputs(
        test_processor, 
        batch_size=1024, 
        model=None,
        actions_to_use=None,
        model_type="critic",
        reduction='mean',
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    model types are critic, actor, value_network. If critic is provided, it will generate q_values and q_values_shuffled.
    """
    
    # get the actions and get their standard deviation
    all_actions = test_processor.rl_dataset["actions"]
    action_dim = all_actions.shape[1]
    
    action_std = torch.as_tensor(all_actions.std(axis=0), dtype=torch.float32).to(device)
    print(f"Action std: {action_std}")    
    
    # loop through trajectories in test_processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(test_processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = test_processor.dynamic_deephit_dataset["stay_ids"]
    
    # create storage
    if model_type == "critic":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)
    elif model_type == "actor":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], action_dim, dtype=torch.float32).to(device)
    elif model_type == "value_network":
        model_outputs = torch.zeros(test_processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)

    for stay_id in stay_ids:
        states, traj_actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(test_processor, traj_start_inds, stay_id, as_tensor=True) 
        
        if actions_to_use is not None:
            # use the provided actions instead of the ones from the dataset
            actions = actions_to_use[indices].to(device)
        else:
            actions = traj_actions
        
        n_batches = int(np.ceil(states.shape[0] / batch_size))
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, states.shape[0])
            states_batch = states[start_idx:end_idx]
            actions_batch = actions[start_idx:end_idx]

            # run inference
            with torch.no_grad():
                if model_type == "critic":
                    q1, q2 = model(states_batch, actions_batch)
                    if reduction == 'mean':
                        model_outputs[indices[start_idx:end_idx]] = ((q1 + q2) / 2.0).flatten()
                    elif reduction == 'min':
                        model_outputs[indices[start_idx:end_idx]] = torch.min(q1, q2).flatten()
                elif model_type == "actor":
                    actor_actions = model(states_batch, deterministic=True)
                    model_outputs[indices[start_idx:end_idx]] = actor_actions[0]
                    # probabilities of actions will have to be implemented here
                elif model_type == "value_network":
                    v_values = model(states_batch)
                    model_outputs[indices[start_idx:end_idx]] = v_values.flatten()

    return model_outputs






def evaluate_multiple_models(rl_model_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parse arguments
    args = parse_args()

    # create configuration
    config_manager = ConfigManager(args)

    # 1. Load the "Jury" (SARSA Critic)
    # Using the exact same architecture as your training
    sarsa_critic, sarsa_configuration, sarsa_hyperparams = load_sarsa_critic(config_manager)

    # load the data
    train_processor, _, test_processor = create_processor_from_shards(sarsa_configuration, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, sarsa_hyperparams)
    test_processor.process_reward(config_manager, sarsa_hyperparams, health_score_median=health_score_median)
    
    # create rl dataset
    batch_size = 1024
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, sarsa_hyperparams)

    # get the model directories
    rl_path = os.path.join(config_manager.paths['rl_models_dir'], config_manager.rl_type, 
                    f"split_{config_manager.split_num}", config_manager.cohort, config_manager.actions,
                    config_manager.reward_type, f"discount_{config_manager.discount}")
    rl_folders = [f for f in os.listdir(rl_path) if os.path.isdir(os.path.join(rl_path, f))]
    if "final_model" in rl_folders or "*_run" in rl_folders:
        rl_folders.remove("final_model")
        rl_folders = [f for f in rl_folders if not f.endswith("_run")]

    sarsa_advantage_results =[]
    rl_advantage_results = []

    for rl_folder in rl_folders:       
        model_path = os.path.join(rl_path, rl_folder)
        
        # load the configuration
        configuration_path = os.path.join(model_path, "config_manager.pkl")
        model_cfg = load_configuration(configuration_path)

        # find directories within model_path that begin with "local" or "unknown"
        model_subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f)) and (f.startswith("local") or f.startswith("unknown"))]

        for subfolder in model_subfolders:
            if rl_model_dir is not None:
                if subfolder != rl_model_dir:
                    continue

            model_subfolder_path = os.path.join(model_path, subfolder)
            
            print(f"Evaluating model in folder: {model_subfolder_path}")

            sarsa_advantage_result, rl_advantage_result = evaluate(
                config_manager=model_cfg, 
                sarsa_critic=sarsa_critic, 
                sarsa_configuration=sarsa_configuration,
                model_path=model_subfolder_path,
                test_processor=test_processor
                )

            sarsa_advantage_results.append(sarsa_advantage_result)
            rl_advantage_results.append(rl_advantage_result)

    # convert results to dataframe and save as csv
    sarsa_advantage_df = pd.DataFrame(sarsa_advantage_results)
    rl_advantage_df = pd.DataFrame(rl_advantage_results)

    sarsa_advantage_df.to_csv(os.path.join(rl_path, 'sarsa_advantage_results.csv'), index=False)
    rl_advantage_df.to_csv(os.path.join(rl_path, 'rl_advantage_results.csv'), index=False)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load terminal sarsa model
    terminal_sarsa_critic, terminal_sarsa_config_manager, terminal_sarsa_hyperparams = load_sarsa_critic(reward_type='terminal')

    # load health_score sarsa model
    health_score_critic, health_score_sarsa_config_manager, health_score_sarsa_hyperparams = load_sarsa_critic(reward_type='health_score_change')

    # load terminal cql model
    terminal_cql_actor, terminal_cql_critic, terminal_cql_config_manager, terminal_cql_hyperparams = load_actor_and_critic()

    # load test data
    test_processor = load_test_data(health_score_sarsa_config_manager, health_score_sarsa_hyperparams)  

    # get and create actions
    clinician_actions = test_processor.rl_dataset["actions"]

    # create zero fluid actions. We don't need zero vaso because that's what the policy outputs
    zero_fluid_actions = torch.zeros_like(clinician_actions)
    zero_fluid_actions[:, 1] = clinician_actions[:, 1]

    # generate policy actions from the terminal cql actor
    cql_actor_actions = generate_model_outputs(
        test_processor=test_processor, 
        batch_size=batch_size, 
        model=terminal_cql_actor,
        model_type="actor",
        device=device
    )

    # confirm the policy is recommending zero vaso
    assert torch.all(cql_actor_actions[:, 1] == 0), "CQL actor is not recommending zero vaso for all states."

    clin_fluid_actor_vaso = clinician_actions.clone()
    clin_fluid_actor_vaso[:, 1] = cql_actor_actions[:, 1]
    actor_fluid_clin_vaso = clinician_actions.clone()
    actor_fluid_clin_vaso[:, 0] = cql_actor_actions[:, 0]

    # generate model outputs for terminal sarsa critic  
    batch_size = health_score_sarsa_hyperparams['batch_size']

    # organize the actions and critics into dictionaries for easy access
    action_variants = {
        "clinician":   clinician_actions,       # baseline
        "zero_vaso":   clin_fluid_actor_vaso,  # == actor's vaso prediction, confirmed near-identical
        "actor_fluid": actor_fluid_clin_vaso,
        "zero_fluid":  zero_fluid_actions,
    }
    
    critics = {
        "cql_terminal": terminal_cql_critic,
        "sarsa_terminal": terminal_sarsa_critic,
        "sarsa_health_score": health_score_critic,
    }

    q_values = {}
    for critic_name, critic in critics.items():
        q_values[critic_name] = {}
        for variant_name, actions in action_variants.items():
            q_values[critic_name][variant_name] = generate_model_outputs(
                test_processor=test_processor,
                batch_size=batch_size,
                model=critic,
                actions_to_use=actions if variant_name != "clinician" else None,
                model_type="critic",
                reduction='mean',
                device=device,
            )

    # calculate advantages for each critic and action variant
    advantages = {}
    for critic_name in critics.keys():
        advantages[critic_name] = {}
        for variant_name in [k for k in action_variants.keys() if k != "clinician"]:
            advantages[critic_name][variant_name] = q_values[critic_name][variant_name] - q_values[critic_name]["clinician"]

    # correlation between advantages and clinician vaso doses by mortality
    mortality = test_processor.rl_dataset["event"]






    pass


if __name__ == "__main__":
    main()