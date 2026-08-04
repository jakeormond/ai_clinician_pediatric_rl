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


SAVE_DIR = "/home/jake/Documents/AI_Clinician/models/analysis/terminal_reward_zero_dose_advantage"
os.makedirs(SAVE_DIR, exist_ok=True)


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


def load_data(config_manager, hyperparams):
    # load the data
    train_processor, val_processor, test_processor = create_processor_from_shards(config_manager, normalize=True, set_to_return='all')

    # process the reward
    health_score_median = train_processor.process_reward(config_manager, hyperparams)
    test_processor.process_reward(config_manager, hyperparams, health_score_median=health_score_median)
    
    # create rl dataset
    train_processor, test_processor, _ = create_test_rl_dataset(train_processor, test_processor, config_manager, hyperparams)

    return train_processor, test_processor


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


def load_sarsa_critic(model_dir=None, shifted_actions=False, reward_type='terminal'):
    if model_dir is None:
        if reward_type == 'terminal':
            sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/terminal_only/final_model_terminal_discount_0.998"
        elif reward_type == 'health_score_change':
            if not shifted_actions:
                sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/health_score_change_only/final_model_discount_0.975"
            else:
                sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/health_score_change_only/final_model_discount_0.975/shifted_actions"
        else:
            raise ValueError(f"Unsupported reward type: {reward_type}")

        inner_directories = [f for f in os.listdir(sarsa_dir) if os.path.isdir(os.path.join(sarsa_dir, f))]
        model_dir = os.path.join(sarsa_dir, inner_directories[0])

        # if inner_directories[0] begins with "local" or "unknown", then we stop
        if not (inner_directories[0].startswith("local") or inner_directories[0].startswith("unknown")):  

            inner_directories = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
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
        configuration_path = os.path.join(model_dir, "config_manager.pkl")
        print(f"Loading config manager from: {configuration_path}")
        # check if config_manager.pkl exists
        if not os.path.exists(configuration_path):
            # then it's in the parent parent directory
            print(f"config_manager.pkl not found in {model_dir}, checking parent directory.")
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
        processor, 
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
    all_actions = processor.rl_dataset["actions"]
    action_dim = all_actions.shape[1]
    
    # action_std = torch.as_tensor(all_actions.std(axis=0), dtype=torch.float32).to(device)
    # print(f"Action std: {action_std}")    
    
    # loop through trajectories in processor.rl_dataset and evaluate the model
    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]
    
    # create storage
    if model_type == "critic":
        model_outputs = torch.zeros(processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)
    elif model_type == "actor":
        model_outputs = torch.zeros(processor.rl_dataset["states"].shape[0], action_dim, dtype=torch.float32).to(device)
    elif model_type == "value_network":
        model_outputs = torch.zeros(processor.rl_dataset["states"].shape[0], dtype=torch.float32).to(device)

    for stay_id in stay_ids:
        states, traj_actions, rewards, next_states, next_actions, dones, mortalities, indices = \
            get_trajectory_data(processor, traj_start_inds, stay_id, as_tensor=True) 
        
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


def get_steps_to_go(processor):
    dones = processor.rl_dataset["dones"]
    traj_end_inds = np.where(dones == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))[:-1]

    steps_to_go = np.zeros(dones.shape[0], dtype=np.int32)
    for start, end in zip(traj_start_inds, traj_end_inds):
        steps_to_go[start:end + 1] = np.arange(end - start, -1, -1)

    return steps_to_go


def _bin_and_average(x, y, n_bins=30, log_x=False):
    """Bins y by x, returns bin centers, means, and SEMs."""
    if log_x:
        bin_edges = np.logspace(0, np.log10(x.max() + 1), n_bins)
        bin_edges = np.concatenate(([0], bin_edges))
    else:
        bin_edges = np.linspace(x.min(), x.max(), n_bins + 1)

    bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, len(bin_edges) - 2)
    centers, means, sems, counts = [], [], [], []
    for b in range(len(bin_edges) - 1):
        in_bin = bin_idx == b
        n = in_bin.sum()
        if n == 0:
            continue
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
        means.append(y[in_bin].mean())
        sems.append(y[in_bin].std() / np.sqrt(n))
        counts.append(n)
    return np.array(centers), np.array(means), np.array(sems), np.array(counts)


def correlation_report(x, y, label):
    """Computes Spearman correlation and returns a formatted string + values."""
    rho, pval = stats.spearmanr(x, y)
    text = f"{label}: ρ={rho:.3f}, p={pval:.2e}, n={len(x)}"
    return rho, pval, text


def compute_signal_penalty_decomposition(
        cql_critic, sarsa_critic, processor,
        zero_action_variant,
        batch_size, device,
        normalize=True,
    ):
    q_cql_real = generate_model_outputs(processor, batch_size, cql_critic, actions_to_use=None, model_type="critic", reduction='mean', device=device)
    q_cql_zero = generate_model_outputs(processor, batch_size, cql_critic, actions_to_use=zero_action_variant, model_type="critic", reduction='mean', device=device)
    q_sarsa_real = generate_model_outputs(processor, batch_size, sarsa_critic, actions_to_use=None, model_type="critic", reduction='mean', device=device)
    q_sarsa_zero = generate_model_outputs(processor, batch_size, sarsa_critic, actions_to_use=zero_action_variant, model_type="critic", reduction='mean', device=device)

    if normalize:
        # scale factor: spread of Q_SARSA over the real, in-distribution dataset actions
        # (computed once per reward arm, using the SARSA critic as the "unpenalized" reference scale)
        scale = q_sarsa_real.std().clamp(min=1e-6)
        q_cql_real, q_cql_zero, q_sarsa_real, q_sarsa_zero = (
            q_cql_real / scale, q_cql_zero / scale, q_sarsa_real / scale, q_sarsa_zero / scale
        )

    signal = q_sarsa_zero - q_sarsa_real                     # note: sign flipped vs. your original — see below
    penalty_real = q_sarsa_real - q_cql_real
    penalty_zero = q_sarsa_zero - q_cql_zero
    delta_penalty = penalty_real - penalty_zero              # differential penalty, not raw penalty
    advantage_cql = q_cql_zero - q_cql_real

    return {
        "signal": signal.cpu().numpy(),
        "delta_penalty": delta_penalty.cpu().numpy(),
        "advantage_cql": advantage_cql.cpu().numpy(),
        "scale_used": scale.item() if normalize else 1.0,
    }




def plot_advantage_analysis(
        advantage,
        steps_to_go,
        mortality,
        real_dose,           # clinician's actual dose for this action dimension
        dose_name,
        critic_name,
        max_steps=None,
        n_bins=30,
        log_x=False,
        save_dir=SAVE_DIR,
        split=None,
        save=True,
    ):
    """
    Two-row figure:
      Row 1: advantage vs. steps_to_go, survival | mortality panels,
             with real clinician dose overlaid on a secondary y-axis
             (to visually check the predicted INVERSE relationship).
      Row 2: advantage vs. real clinician dose directly, survival | mortality.
    Spearman correlations reported for both relationships, both panels.
    """
    advantage = np.asarray(advantage)
    steps_to_go = np.asarray(steps_to_go)
    mortality = np.asarray(mortality)
    real_dose = np.asarray(real_dose)

    if max_steps is not None:
        mask = steps_to_go <= max_steps
        advantage, steps_to_go, mortality, real_dose = (
            advantage[mask], steps_to_go[mask], mortality[mask], real_dose[mask]
        )

    survival_mask = mortality == 0
    mortality_mask = mortality == 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    group_info = [
        (0, survival_mask, "Survival", "seagreen"),
        (1, mortality_mask, "Mortality", "crimson"),
    ]

    report_lines = []

    for col, mask, label, color in group_info:
        adv = advantage[mask]
        steps = steps_to_go[mask]
        dose = real_dose[mask]

        # --- Row 1: advantage vs steps_to_go, with real dose overlaid ---
        ax1 = axes[0, col]
        centers, means, sems, _ = _bin_and_average(steps, adv, n_bins, log_x)
        dose_centers, dose_means, dose_sems, _ = _bin_and_average(steps, dose, n_bins, log_x)

        rho1, p1, text1 = correlation_report(steps, adv, f"{label} — advantage vs steps_to_go")
        report_lines.append(text1)

        ax1.plot(centers, means, color=color, lw=2, label=f"{dose_name} advantage")
        ax1.fill_between(centers, means - sems, means + sems, color=color, alpha=0.25)
        ax1.axhline(0, color='gray', lw=1, ls='--')
        ax1.set_ylabel(f"Advantage: Q(counterfactual) - Q(real)")
        ax1.set_title(f"{label}  (ρ={rho1:.3f}, p={p1:.1e})")
        ax1.invert_xaxis()
        if log_x:
            ax1.set_xscale('log')

        ax1b = ax1.twinx()
        ax1b.plot(dose_centers, dose_means, color='black', lw=1.5, ls=':', alpha=0.6, label=f"real {dose_name} dose")
        ax1b.set_ylabel(f"Real clinician {dose_name} dose", color='gray')
        ax1b.tick_params(axis='y', colors='gray')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1b, labels1b = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='best', fontsize=8)

        # --- Row 2: advantage vs real dose directly ---
        ax2 = axes[1, col]
        centers2, means2, sems2, _ = _bin_and_average(dose, adv, n_bins=20, log_x=False)

        rho2, p2, text2 = correlation_report(dose, adv, f"{label} — advantage vs real dose")
        report_lines.append(text2)

        ax2.plot(centers2, means2, color=color, lw=2)
        ax2.fill_between(centers2, means2 - sems2, means2 + sems2, color=color, alpha=0.25)
        ax2.axhline(0, color='gray', lw=1, ls='--')
        ax2.set_xlabel(f"Real clinician {dose_name} dose")
        ax2.set_ylabel("Advantage")
        ax2.set_title(f"{label}  (ρ={rho2:.3f}, p={p2:.1e})")

    axes[0, 0].set_xlabel("Steps remaining until trajectory end")
    axes[0, 1].set_xlabel("Steps remaining until trajectory end")

    fig.suptitle(f"{critic_name}: {dose_name} advantage analysis")
    plt.tight_layout()

    if save:
        filename = f"{critic_name}_{dose_name}_advantage_analysis.png"
        # check if split is None, if so, save in SAVE_DIR, else save in SAVE_DIR/split
        if split is not None:
            save_dir = os.path.join(save_dir, split)
            os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=200, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if not save:
        plt.show()
    plt.close(fig)  # prevent figures from accumulating in memory across the loop

    print("\n".join(report_lines))

    return report_lines


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load terminal sarsa model
    terminal_sarsa_critic, terminal_sarsa_config_manager, terminal_sarsa_hyperparams = load_sarsa_critic(reward_type='terminal')

    # load health_score sarsa model
    health_score_critic, health_score_sarsa_config_manager, health_score_sarsa_hyperparams = load_sarsa_critic(reward_type='health_score_change')

    # load terminal cql model
    terminal_cql_actor, terminal_cql_critic, terminal_cql_config_manager, terminal_cql_hyperparams = load_actor_and_critic()

    # load test data
    processors = {}
    processors['train'], processors['test'] = load_data(terminal_sarsa_config_manager, terminal_sarsa_hyperparams)  

    for split, processor in processors.items():
        # get and create actions
        clinician_actions = processor.rl_dataset["actions"]

        # create zero fluid actions. We don't need zero vaso because that's what the policy outputs
        zero_fluid_actions = torch.zeros_like(clinician_actions)
        zero_fluid_actions[:, 1] = clinician_actions[:, 1]

        # get mortality
        mortality = processor.rl_dataset["mortality"]

        # calculate steps to go
        steps_to_go = get_steps_to_go(processor)

        ### calculate correlations and plot for each critic and action variant ###############
        clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else clinician_actions
        mortality_np = mortality.cpu().numpy() if torch.is_tensor(mortality) else mortality
        steps_to_go_np = steps_to_go.cpu().numpy() if torch.is_tensor(steps_to_go) else steps_to_go

        dose_dim = {"zero_vaso": 1, "actor_fluid": 0, "zero_fluid": 0}  # action index per variant

        # generate policy actions from the terminal cql actor
        batch_size = health_score_sarsa_hyperparams['batch_size']
        cql_actor_actions = generate_model_outputs(
            processor=processor, 
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
                    processor=processor,
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

        all_reports = {}
        for critic_name in critics.keys():
            all_reports[critic_name] = {}
            for variant_name in ["zero_vaso", "actor_fluid", "zero_fluid"]:
                adv = advantages[critic_name][variant_name]
                adv_np = adv.cpu().numpy() if torch.is_tensor(adv) else adv

                dim = dose_dim[variant_name]
                real_dose = clinician_actions_np[:, dim]

                report = plot_advantage_analysis(
                    advantage=adv_np,
                    steps_to_go=steps_to_go_np,
                    mortality=mortality_np,
                    real_dose=real_dose,
                    dose_name=variant_name,
                    critic_name=critic_name,
                    save=True,
                    split=split
                )
                all_reports[critic_name][variant_name] = report






    pass


if __name__ == "__main__":
    main()