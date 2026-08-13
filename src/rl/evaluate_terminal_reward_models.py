import argparse
import json
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
import seaborn as sns
import time

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
from sklearn.metrics import roc_auc_score

from ..setup_environment import load_paths


SURVIVE_COLOR = "#2471a3"
DIE_COLOR = "#c0392b"

TERMINAL_REWARD_COLOR = "#7d3c98"   # purple
HEALTH_SCORE_REWARD_COLOR = "#229954"  # green

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
    actions_shift = hyperparams['actions_shift'] 

    scaler = train_processor.create_rl_dataset(
        states="model_outputs", 
        scaler=None, 
        action_names=action_names, 
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma,
        actions_shift=actions_shift
    ) 

    _ = test_processor.create_rl_dataset(
        states="model_outputs",
        scaler=scaler,
        action_names=action_names,
        use_compressed_latents=use_compressed_latents,
        n_step=n_step,
        gamma=gamma,
        actions_shift=actions_shift
    )

    return train_processor, test_processor, scaler


def load_sarsa_critic(model_dir=None, shifted_actions=True, reward_type='terminal'):
    if model_dir is None:
        if reward_type == 'terminal':
            sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/terminal_only/final_model_terminal_discount_0.998"
        elif reward_type == 'health_score_change':
            if not shifted_actions:
                sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/health_score_change_only/final_model_discount_0.975"
            else:
                sarsa_dir = "/home/jake/Documents/AI_Clinician/models/rl_models/choc/basic_q/split_0/sepsis/intake-vis/health_score_change_only/final_model_discount_0.975/shifted_actions/vaso_reward_positive_control/vaso_coeff_0.2"
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
        raise FileNotFoundError(f"results.json not found in {model_dir}")
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



SURVIVE_COLOR = "#2471a3"
DIE_COLOR = "#c0392b"
 
 
def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
 
 
def _build_long_df(q_values_list, mortality):
    """
    q_values_list: list of n_seeds arrays, each (n_test,)
    mortality: (n_test,) array, shared across all seeds (same test set)
 
    Returns a long-format dataframe: columns = ['seed', 'q_value', 'outcome']
    """
    mortality = _to_numpy(mortality)
    rows = []
    for seed_idx, q in enumerate(q_values_list):
        q = _to_numpy(q).reshape(-1)
        if q.shape[0] != mortality.shape[0]:
            raise ValueError(
                f"Seed {seed_idx}: q_values length ({q.shape[0]}) != mortality length "
                f"({mortality.shape[0]}). Are these from the same test set / ordering?"
            )
        outcome = np.where(mortality == 1, "Non-survivor", "Survivor")
        rows.append(pd.DataFrame({
            "seed": f"Seed {seed_idx + 1}",
            "q_value": q,
            "outcome": outcome,
        }))
    return pd.concat(rows, ignore_index=True)
 
 
def plot_q_by_outcome(q_values_list, mortality, title, save_path):
    df = _build_long_df(q_values_list, mortality)
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    sns.violinplot(
        data=df, x="seed", y="q_value", hue="outcome",
        split=True, inner="quartile", ax=ax,
        palette={"Survivor": SURVIVE_COLOR, "Non-survivor": DIE_COLOR},
        cut=0, linewidth=1,
    )
 
    ax.set_xlabel("")
    ax.set_ylabel("Q-value (real clinician actions)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title=None, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    return fig, df



def _seed_means(q_values_list, mortality):
    """Returns (survivor_means, non_survivor_means), each length n_seeds."""
    mortality = _to_numpy(mortality)
    survivor_means, non_survivor_means = [], []
    for q in q_values_list:
        q = _to_numpy(q).reshape(-1)
        survivor_means.append(q[mortality == 0].mean())
        non_survivor_means.append(q[mortality == 1].mean())
    return np.array(survivor_means), np.array(non_survivor_means)
 
 
def plot_mean_q_by_outcome(
    terminal_q_values,
    health_score_q_values,
    mortality,
    save_path="/mnt/user-data/outputs/mean_q_by_outcome.png",
):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
 
    conditions = [
        ("Terminal reward", terminal_q_values, axes[0]),
        ("Health-score-difference reward", health_score_q_values, axes[1]),
    ]
 
    rng = np.random.default_rng(0)  # fixed seed so jitter is reproducible across re-runs
 
    for title, q_values_list, ax in conditions:
        surv, non_surv = _seed_means(q_values_list, mortality)
        n_seeds = len(surv)
        x_surv, x_non_surv = 0, 1
 
        # small per-seed jitter (same jitter offset used for both the point and its
        # connecting line, so overlapping seeds — common for terminal — become
        # visible as separate points/lines rather than stacking exactly on top of each other)
        jitter = rng.uniform(-0.06, 0.06, size=n_seeds)
        x_surv_j = x_surv + jitter
        x_non_surv_j = x_non_surv + jitter
 
        # thin connecting line per seed
        for xs, xn, s, ns in zip(x_surv_j, x_non_surv_j, surv, non_surv):
            ax.plot([xs, xn], [s, ns], color="grey", alpha=0.5, linewidth=1, zorder=1)
 
        ax.scatter(x_surv_j, surv, color=SURVIVE_COLOR, s=60, zorder=2, label="Survivor")
        ax.scatter(x_non_surv_j, non_surv, color=DIE_COLOR, s=60, zorder=2, label="Non-survivor")
 
        ax.set_xticks([x_surv, x_non_surv])
        ax.set_xticklabels(["Survivor", "Non-survivor"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("Mean Q-value (real clinician actions)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
 
    fig.suptitle(f"Mean Q-value by outcome, per seed (n={len(terminal_q_values)} seeds/condition)", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close(fig)  # prevent figures from accumulating in memory across the loop
    return fig
 


def _beeswarm_offsets(values, min_sep, max_offset=0.18, step=0.045):
    """
    Deterministic x-offsets for a 1D beeswarm: points that are close together
    in value get pushed apart symmetrically; points that are already far from
    their neighbors stay at offset 0. Not random — same input always gives
    the same layout, and separation reflects actual closeness in value.
 
    values: 1D array of y-values (e.g. AUCs) to place along a single column.
    min_sep: values closer than this (in y-units) are considered "colliding"
             and will be pushed apart.
    max_offset / step: how far apart collision resolution can push points,
             and the increment used per collision, in x-axis units.
    """
    order = np.argsort(values)
    offsets = np.zeros(len(values))
    placed = []  # (y_value, x_offset) of points already placed
 
    for idx in order:
        y = values[idx]
        colliding = [p for p in placed if abs(p[0] - y) < min_sep]
        if not colliding:
            offsets[idx] = 0.0
        else:
            k = len(colliding)
            side = 1 if k % 2 == 1 else -1
            magnitude = min(((k + 1) // 2) * step, max_offset)
            offsets[idx] = side * magnitude
        placed.append((y, offsets[idx]))
 
    return offsets
 
 
def _seed_aucs(q_values_list, mortality):
    """Returns array of AUCs, one per seed, using each model's full test-set Q values."""
    mortality = _to_numpy(mortality)
    aucs = []
    for q in q_values_list:
        q = _to_numpy(q).reshape(-1)
        auc = roc_auc_score(1 - mortality, q)  # higher Q -> predicts survival
        aucs.append(auc)
    return np.array(aucs)
 
 
def plot_auc_by_condition(
    terminal_q_values,
    health_score_q_values,
    mortality,
    save_path="/mnt/user-data/outputs/auc_by_condition.png",
):
    auc_terminal = _seed_aucs(terminal_q_values, mortality)
    auc_health = _seed_aucs(health_score_q_values, mortality)
 
    fig, ax = plt.subplots(figsize=(5.5, 5))
 
    # min_sep controls how close two AUC values need to be before they're
    # pushed apart — set relative to the full spread across both conditions
    # so it adapts to whatever scale your actual AUCs land on (e.g. terminal
    # spanning 0.6-0.9 vs. health-score clustered tightly near 0.5).
    all_aucs = np.concatenate([auc_terminal, auc_health])
    min_sep = max((all_aucs.max() - all_aucs.min()) * 0.03, 1e-4)
 
    x_terminal = 0 + _beeswarm_offsets(auc_terminal, min_sep=min_sep)
    x_health = 1 + _beeswarm_offsets(auc_health, min_sep=min_sep)
 
    ax.scatter(x_terminal, auc_terminal, color=TERMINAL_REWARD_COLOR, s=70, alpha=0.85,
               edgecolors="white", linewidths=0.8, zorder=3, label="Terminal")
    ax.scatter(x_health, auc_health, color=HEALTH_SCORE_REWARD_COLOR, s=70, alpha=0.85,
               edgecolors="white", linewidths=0.8, zorder=3, label="Health-score diff.")
 
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, zorder=1)
    ax.text(1.32, 0.5, "chance", va="center", fontsize=9, color="grey")
 
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Terminal\nreward", "Health-score\ndifference"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("AUC (Q-value as survival predictor)")
    ax.set_title("Q-value discrimination of survival, per seed\n(n=5 seeds/condition)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    print(f"Terminal AUC:     mean={auc_terminal.mean():.3f}, values={np.round(auc_terminal, 3)}")
    print(f"Health-score AUC: mean={auc_health.mean():.3f}, values={np.round(auc_health, 3)}")
    return fig, auc_terminal, auc_health
 




def action_sensitivity_sweep(
        processor,
        model,
        clinician_actions,
        sweep_dim,          # 0 = fluid, 1 = vaso — which action to sweep
        n_grid=20,
        batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    Sweeps `sweep_dim` across [0, 1], holding the OTHER action dimension at
    the clinician's real dose throughout (since swept_actions is cloned from
    clinician_actions and only sweep_dim is overwritten). Use sweep_dim=1
    for vaso, sweep_dim=0 for fluid.
    """
    grid = torch.linspace(0, 1, n_grid)
    q_at_grid = []
 
    for grid_idx, val in enumerate(grid):
        print(f"    grid point {grid_idx + 1}/{n_grid} (value={val.item():.3f})...", flush=True)
        swept_actions = clinician_actions.clone()
        swept_actions[:, sweep_dim] = val
        # the other dimension is untouched -> stays at clinician's real dose
 
        q_vals = generate_model_outputs(
            processor,
            model=model,
            actions_to_use=swept_actions,
            model_type="critic",
            reduction='mean',
            batch_size=batch_size,
            device=device,
        )
        q_at_grid.append(q_vals)
 
    q_at_grid = torch.stack(q_at_grid, dim=1)  # (n_timesteps, n_grid)
    q_range = q_at_grid.max(dim=1).values - q_at_grid.min(dim=1).values
 
    return {
        "q_at_grid": q_at_grid,
        "q_range": q_range,
        "grid": grid,
    }
 
 
def compute_seed_sensitivities(models, q_values_list, processor, clinician_actions, sweep_dim, dim_label="", condition_label=""):
    """
    Runs the sweep for every seed's model, returns one median normalized
    sensitivity (% of that seed's own real-action Q IQR) per seed.
    """
    import time
 
    medians = []
    n_seeds = len(models)
    for seed_idx, (model_entry, q_real) in enumerate(zip(models, q_values_list)):
        t0 = time.time()
        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: starting sweep...", flush=True)
        model = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
        sweep = action_sensitivity_sweep(processor, model, clinician_actions, sweep_dim=sweep_dim)
        pct = normalize_sensitivity(sweep["q_range"], q_real)
        medians.append(np.median(pct))
        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: done in {time.time() - t0:.1f}s "
              f"(median sensitivity = {np.median(pct):.1f}%)", flush=True)
    return np.array(medians)
 
 
def _beeswarm_offsets(values, min_sep, max_offset=0.18, step=0.045):
    """Deterministic beeswarm x-offsets — see plot_auc_by_condition.py for details."""
    order = np.argsort(values)
    offsets = np.zeros(len(values))
    placed = []
    for idx in order:
        y = values[idx]
        colliding = [p for p in placed if abs(p[0] - y) < min_sep]
        if not colliding:
            offsets[idx] = 0.0
        else:
            k = len(colliding)
            side = 1 if k % 2 == 1 else -1
            magnitude = min(((k + 1) // 2) * step, max_offset)
            offsets[idx] = side * magnitude
        placed.append((y, offsets[idx]))
    return offsets
 
 
def plot_action_sensitivity_by_condition(
    terminal_models, health_score_models,
    terminal_q_values, health_score_q_values,
    processor, clinician_actions,
    save_path="/mnt/user-data/outputs/action_sensitivity_by_condition.png",
):
    """
    Two panels (vaso, fluid), each showing terminal vs. health-score
    median-per-seed normalized sensitivity, same beeswarm/color style as
    plot_auc_by_condition. Magnitude only — no dosing direction/preference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
 
    dims = [("Vasopressor", 1, axes[0]), ("Fluid", 0, axes[1])]
    results = {}
 
    for dim_label, sweep_dim, ax in dims:
        print(f"\n=== {dim_label} sweep ===", flush=True)
 
        print(f"-- Terminal reward models --", flush=True)
        sens_terminal = compute_seed_sensitivities(
            terminal_models, terminal_q_values, processor, clinician_actions, sweep_dim,
            dim_label=dim_label, condition_label="terminal"
        )
        print(f"-- Health-score-difference models --", flush=True)
        sens_health = compute_seed_sensitivities(
            health_score_models, health_score_q_values, processor, clinician_actions, sweep_dim,
            dim_label=dim_label, condition_label="health_score"
        )
        results[dim_label] = {"terminal": sens_terminal, "health_score": sens_health}
 
        all_vals = np.concatenate([sens_terminal, sens_health])
        min_sep = max((all_vals.max() - all_vals.min()) * 0.03, 1e-4)
 
        x_terminal = 0 + _beeswarm_offsets(sens_terminal, min_sep=min_sep)
        x_health = 1 + _beeswarm_offsets(sens_health, min_sep=min_sep)
 
        ax.scatter(x_terminal, sens_terminal, color=TERMINAL_REWARD_COLOR, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3, label="Terminal")
        ax.scatter(x_health, sens_health, color=HEALTH_SCORE_REWARD_COLOR, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=3, label="Health-score diff.")
 
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Terminal\nreward", "Health-score\ndifference"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("Median action-sweep Q range\n(% of state-wise Q IQR)")
        ax.set_title(f"{dim_label} dose sensitivity\n(n={len(sens_terminal)} seeds/condition)",
                     fontsize=11, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
 
        print(f"{dim_label} — Terminal:     mean={sens_terminal.mean():.1f}%, values={np.round(sens_terminal, 1)}")
        print(f"{dim_label} — Health-score: mean={sens_health.mean():.1f}%, values={np.round(sens_health, 1)}")
 
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    return fig, results
def normalize_sensitivity(q_range, reference_q_real):
    """
    Expresses per-timestep action-sensitivity (q_range) as a percentage
    of the reference critic's overall between-state Q spread (IQR),
    computed at real clinician actions — puts sensitivity on a comparable
    scale across critics/reward arms with very different absolute Q scales.
    """
    q25, q75 = torch.quantile(reference_q_real, 0.25), torch.quantile(reference_q_real, 0.75)
    ref_iqr = (q75 - q25).clamp(min=1e-6)
    return (q_range / ref_iqr * 100).cpu().numpy()



"""
Occupancy-partitioned action-sensitivity.
 
For each timestep's action-sensitivity sweep, restricts the max-min Q-range
calculation to only the grid points whose (fluid, vaso) combination has real
data support in the training set (occupancy count >= min_count), using the
same 2D joint-occupancy binning as build_zero_occupancy_action_pool. Compares
this "well-supported only" sensitivity against the original unrestricted
sensitivity.
 
Rationale: if health-score's higher sensitivity survives restriction to
well-supported (state-independent) action-space regions, that argues the
effect is learned signal rather than smooth extrapolation into sparse/empty
regions of action space.
 
Expects, already in your session:
    terminal_models, health_score_models       : lists of 5 loaded critics
    terminal_q_values, health_score_q_values   : lists of 5 Q-value tensors at
        real clinician actions (for the IQR normalization reference)
    processors['test']                          : test processor
    processors['train'] (or wherever your training actions live)          
    clinician_actions                           : test-set clinician actions, (n, 2)
    generate_model_outputs, normalize_sensitivity, action_sensitivity_sweep : your existing functions
"""
 

def build_joint_occupancy_counts(actions, n_bins=20):
    """
    Full 2D joint occupancy histogram of (fluid, vaso) actions, plus bin
    edges — same binning convention as build_zero_occupancy_action_pool,
    but returns the full count grid rather than just the empty-bin list.
    Should be built from TRAINING data (what the critic actually learned
    from), not test data.
    """
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    joint_counts, fluid_edges, vaso_edges = np.histogram2d(
        actions_np[:, 0], actions_np[:, 1], bins=n_bins
    )
    return joint_counts, fluid_edges, vaso_edges
 
 
def _bin_index(values, edges):
    """0-indexed bin index per value, matching np.histogram2d's convention.
    Values outside [edges[0], edges[-1]] are clipped into the nearest edge bin."""
    idx = np.digitize(values, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)
 
 
def occupancy_partitioned_range(
        q_at_grid,        # (n_timesteps, n_grid) from action_sensitivity_sweep
        grid,              # (n_grid,) swept values
        real_other_dose,   # (n_timesteps,) real clinician dose for the OTHER (held-fixed) action
        sweep_dim,         # 0 = fluid swept, 1 = vaso swept
        joint_counts, fluid_edges, vaso_edges,
        min_count=10,
    ):
    """
    Returns, per timestep: full (unrestricted) Q-range, occupancy-restricted
    Q-range (NaN if fewer than 2 well-supported grid points), and the
    fraction of grid points that were well-supported.
    """
    q = q_at_grid.cpu().numpy() if torch.is_tensor(q_at_grid) else np.asarray(q_at_grid)
    grid_np = grid.cpu().numpy() if torch.is_tensor(grid) else np.asarray(grid)
    real_other_np = real_other_dose.cpu().numpy() if torch.is_tensor(real_other_dose) else np.asarray(real_other_dose)
 
    n_timesteps, n_grid = q.shape
 
    if sweep_dim == 1:  # vaso swept, fluid held at real dose
        fluid_vals = np.repeat(real_other_np[:, None], n_grid, axis=1)
        vaso_vals = np.tile(grid_np[None, :], (n_timesteps, 1))
    else:  # fluid swept, vaso held at real dose
        fluid_vals = np.tile(grid_np[None, :], (n_timesteps, 1))
        vaso_vals = np.repeat(real_other_np[:, None], n_grid, axis=1)
 
    fluid_bin = _bin_index(fluid_vals, fluid_edges)
    vaso_bin = _bin_index(vaso_vals, vaso_edges)
    counts = joint_counts[fluid_bin, vaso_bin]
    well_supported = counts >= min_count
 
    full_range = q.max(axis=1) - q.min(axis=1)
    restricted_range = np.full(n_timesteps, np.nan)
    for t in range(n_timesteps):
        mask = well_supported[t]
        if mask.sum() >= 2:
            vals = q[t, mask]
            restricted_range[t] = vals.max() - vals.min()
 
    frac_supported = well_supported.mean(axis=1)
 
    return {
        "full_range": full_range,
        "restricted_range": restricted_range,
        "frac_supported": frac_supported,
    }
 
 
def build_joint_state_action_occupancy_counts(health_score, actions, n_severity_bins=5, n_action_bins=10):
    """
    3D joint occupancy histogram: severity (health score) x fluid x vaso,
    built from TRAINING data. Coarser action binning (default 10x10 rather
    than 20x20) than the marginal 2D histogram, since adding a third
    dimension thins out cell counts fast — kept coarse enough that cells
    stay meaningfully populated.
 
    Severity bins are QUANTILES of the training health-score distribution,
    not fixed-width — health score is unlikely to be uniformly distributed
    across 0-100, so quantile bins keep roughly equal numbers of training
    timesteps per severity bin.
 
    Returns: joint_counts (n_severity_bins, n_action_bins, n_action_bins),
             severity_edges, fluid_edges, vaso_edges
    """
    health_score_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
 
    severity_edges = np.quantile(health_score_np, np.linspace(0, 1, n_severity_bins + 1))
    severity_edges[0], severity_edges[-1] = -np.inf, np.inf  # catch any out-of-training-range test values
 
    fluid_edges = np.linspace(0, 1, n_action_bins + 1)
    vaso_edges = np.linspace(0, 1, n_action_bins + 1)
 
    severity_bin = _bin_index(health_score_np, severity_edges)
    fluid_bin = _bin_index(actions_np[:, 0], fluid_edges)
    vaso_bin = _bin_index(actions_np[:, 1], vaso_edges)
 
    joint_counts = np.zeros((n_severity_bins, n_action_bins, n_action_bins))
    np.add.at(joint_counts, (severity_bin, fluid_bin, vaso_bin), 1)
 
    return joint_counts, severity_edges, fluid_edges, vaso_edges
 
 
def occupancy_partitioned_range_state_aware(
        q_at_grid,          # (n_timesteps, n_grid)
        grid,                # (n_grid,)
        real_other_dose,     # (n_timesteps,)
        sweep_dim,
        test_health_score,   # (n_timesteps,) — THIS test set's own severity, used to pick each
                              # timestep's own severity bin (built from training-set quantile edges)
        joint_counts, severity_edges, fluid_edges, vaso_edges,
        min_count=10,
    ):
    """
    Same as occupancy_partitioned_range, but occupancy is looked up within
    each timestep's OWN severity bin, rather than pooled across the whole
    training set — i.e. "was this dose combination seen in patients this
    sick," not "was this dose combination seen anywhere."
    """
    q = q_at_grid.cpu().numpy() if torch.is_tensor(q_at_grid) else np.asarray(q_at_grid)
    grid_np = grid.cpu().numpy() if torch.is_tensor(grid) else np.asarray(grid)
    real_other_np = real_other_dose.cpu().numpy() if torch.is_tensor(real_other_dose) else np.asarray(real_other_dose)
    health_score_np = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
 
    n_timesteps, n_grid = q.shape
    severity_bin = _bin_index(health_score_np, severity_edges)  # (n_timesteps,) — fixed per timestep, doesn't vary across the sweep
 
    if sweep_dim == 1:  # vaso swept, fluid held at real dose
        fluid_vals = np.repeat(real_other_np[:, None], n_grid, axis=1)
        vaso_vals = np.tile(grid_np[None, :], (n_timesteps, 1))
    else:
        fluid_vals = np.tile(grid_np[None, :], (n_timesteps, 1))
        vaso_vals = np.repeat(real_other_np[:, None], n_grid, axis=1)
 
    fluid_bin = _bin_index(fluid_vals, fluid_edges)
    vaso_bin = _bin_index(vaso_vals, vaso_edges)
    severity_bin_broadcast = np.repeat(severity_bin[:, None], n_grid, axis=1)
 
    counts = joint_counts[severity_bin_broadcast, fluid_bin, vaso_bin]
    well_supported = counts >= min_count
 
    full_range = q.max(axis=1) - q.min(axis=1)
    restricted_range = np.full(n_timesteps, np.nan)
    for t in range(n_timesteps):
        mask = well_supported[t]
        if mask.sum() >= 2:
            vals = q[t, mask]
            restricted_range[t] = vals.max() - vals.min()
 
    frac_supported = well_supported.mean(axis=1)
 
    return {
        "full_range": full_range,
        "restricted_range": restricted_range,
        "frac_supported": frac_supported,
    }
 
 
def compute_seed_occupancy_sensitivities(
        models, q_values_list, processor, clinician_actions, sweep_dim,
        joint_counts, fluid_edges, vaso_edges, min_count=10,
        dim_label="", condition_label="",
        state_aware=False, test_health_score=None, severity_edges=None,
    ):
    """
    Runs the sweep + occupancy partition for every seed's model. Returns
    (full_medians, restricted_medians, coverage) arrays, each length n_seeds.
    restricted_medians uses only timesteps with a valid (non-NaN) restricted
    range; coverage = fraction of timesteps with >=2 well-supported grid points.
 
    If state_aware=True, uses occupancy_partitioned_range_state_aware (severity
    x fluid x vaso occupancy) instead of the marginal (fluid x vaso only)
    version — requires test_health_score and severity_edges, and joint_counts
    should be the 3D array from build_joint_state_action_occupancy_counts.
    """
    other_dim = 0 if sweep_dim == 1 else 1
    real_other_dose = clinician_actions[:, other_dim]
 
    full_medians, restricted_medians, coverage, frac_supported_list = [], [], [], []
    n_seeds = len(models)
 
    for seed_idx, (model_entry, q_real) in enumerate(zip(models, q_values_list)):
        t0 = time.time()
        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: running sweep + occupancy partition...", flush=True)
        model = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
 
        sweep = action_sensitivity_sweep(processor, model, clinician_actions, sweep_dim=sweep_dim)
 
        if state_aware:
            part = occupancy_partitioned_range_state_aware(
                sweep["q_at_grid"], sweep["grid"], real_other_dose, sweep_dim,
                test_health_score, joint_counts, severity_edges, fluid_edges, vaso_edges,
                min_count=min_count,
            )
        else:
            part = occupancy_partitioned_range(
                sweep["q_at_grid"], sweep["grid"], real_other_dose, sweep_dim,
                joint_counts, fluid_edges, vaso_edges, min_count=min_count,
            )
 
        full_pct = normalize_sensitivity(
            torch.as_tensor(part["full_range"]), q_real.detach().cpu()
        )
        valid = ~np.isnan(part["restricted_range"])
        restricted_pct = normalize_sensitivity(
            torch.as_tensor(part["restricted_range"][valid]), q_real.detach().cpu()
        ) if valid.sum() > 0 else np.array([np.nan])
 
        full_medians.append(np.median(full_pct))
        restricted_medians.append(np.median(restricted_pct))
        cov = valid.mean()  # fraction of timesteps where a restricted range could even be computed
        coverage.append(cov)
        mean_frac_supported = part["frac_supported"].mean()  # avg fraction of the 20 grid points that were well-supported, per timestep
        frac_supported_list.append(mean_frac_supported)
 
        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: done in {time.time() - t0:.1f}s "
              f"(full median={np.median(full_pct):.1f}%, restricted median={np.median(restricted_pct):.1f}%, "
              f"coverage={cov*100:.0f}% of timesteps had >=2 well-supported grid points, "
              f"avg {mean_frac_supported*100:.0f}% of the 20 grid points were well-supported per timestep)", flush=True)
 
    return np.array(full_medians), np.array(restricted_medians), np.array(coverage), np.array(frac_supported_list)
 
 
def _beeswarm_offsets(values, min_sep, max_offset=0.18, step=0.045):
    order = np.argsort(values)
    offsets = np.zeros(len(values))
    placed = []
    for idx in order:
        y = values[idx]
        colliding = [p for p in placed if abs(p[0] - y) < min_sep]
        if not colliding:
            offsets[idx] = 0.0
        else:
            k = len(colliding)
            side = 1 if k % 2 == 1 else -1
            magnitude = min(((k + 1) // 2) * step, max_offset)
            offsets[idx] = side * magnitude
        placed.append((y, offsets[idx]))
    return offsets
 
 
def plot_occupancy_partitioned_sensitivity(
    terminal_models, health_score_models,
    terminal_q_values, health_score_q_values,
    processor, clinician_actions,
    training_actions,          # actions from the TRAINING set, for building occupancy histogram
    min_count=10,
    save_path="/mnt/user-data/outputs/occupancy_partitioned_sensitivity.png",
    state_aware=False,
    training_health_score=None,   # required if state_aware=True — training-set health scores
    test_health_score=None,       # required if state_aware=True — test-set health scores
    n_severity_bins=5,
    n_action_bins=10,
):
    if state_aware:
        if training_health_score is None or test_health_score is None:
            raise ValueError("state_aware=True requires both training_health_score and test_health_score")
        joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
            training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
        )
        restriction_label = f"Joint state+dose\nsupported (n≥{min_count})"
        title_suffix = "restricted to joint severity+dose data-supported regions"
    else:
        joint_counts, fluid_edges, vaso_edges = build_joint_occupancy_counts(training_actions)
        severity_edges = None
        restriction_label = f"Well-supported\nonly (n≥{min_count})"
        title_suffix = "restricted to data-supported regions"
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    dims = [("Vasopressor", 1, axes[0]), ("Fluid", 0, axes[1])]
    results = {}
 
    for dim_label, sweep_dim, ax in dims:
        print(f"\n=== {dim_label}: occupancy-partitioned sweep ({'state-aware' if state_aware else 'marginal'}) ===", flush=True)
 
        print("-- Terminal reward models --", flush=True)
        full_t, restr_t, cov_t, frac_t = compute_seed_occupancy_sensitivities(
            terminal_models, terminal_q_values, processor, clinician_actions, sweep_dim,
            joint_counts, fluid_edges, vaso_edges, min_count=min_count,
            dim_label=dim_label, condition_label="terminal",
            state_aware=state_aware, test_health_score=test_health_score, severity_edges=severity_edges,
        )
        print("-- Health-score-difference models --", flush=True)
        full_h, restr_h, cov_h, frac_h = compute_seed_occupancy_sensitivities(
            health_score_models, health_score_q_values, processor, clinician_actions, sweep_dim,
            joint_counts, fluid_edges, vaso_edges, min_count=min_count,
            dim_label=dim_label, condition_label="health_score",
            state_aware=state_aware, test_health_score=test_health_score, severity_edges=severity_edges,
        )
 
        results[dim_label] = {
            "terminal": {"full": full_t, "restricted": restr_t, "coverage": cov_t, "frac_supported": frac_t},
            "health_score": {"full": full_h, "restricted": restr_h, "coverage": cov_h, "frac_supported": frac_h},
        }
 
        # paired points: x=0 "Full range", x=1 "Well-supported only", one line per seed
        rng_all = np.concatenate([full_t, restr_t, full_h, restr_h])
        min_sep = max((np.nanmax(rng_all) - np.nanmin(rng_all)) * 0.03, 1e-4)
 
        for full_vals, restr_vals, color, label in [
            (full_t, restr_t, TERMINAL_REWARD_COLOR, "Terminal"),
            (full_h, restr_h, HEALTH_SCORE_REWARD_COLOR, "Health-score diff."),
        ]:
            x_full = 0 + _beeswarm_offsets(full_vals, min_sep=min_sep)
            x_restr = 1 + _beeswarm_offsets(restr_vals, min_sep=min_sep)
            for xf, xr, fv, rv in zip(x_full, x_restr, full_vals, restr_vals):
                ax.plot([xf, xr], [fv, rv], color=color, alpha=0.4, linewidth=1, zorder=1)
            ax.scatter(x_full, full_vals, color=color, s=60, alpha=0.9, edgecolors="white",
                       linewidths=0.7, zorder=2, label=label)
            ax.scatter(x_restr, restr_vals, color=color, s=60, alpha=0.9, edgecolors="white",
                       linewidths=0.7, zorder=2)
 
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Full sweep\nrange", restriction_label])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("Median normalized sensitivity\n(% of state-wise Q IQR)")
        ax.set_title(f"{dim_label}", fontsize=11, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
 
        print(f"{dim_label} — Terminal:     full={full_t.mean():.1f}% (coverage={cov_t.mean()*100:.0f}%, "
              f"avg grid-point support={frac_t.mean()*100:.0f}%), restricted={restr_t.mean():.1f}%")
        print(f"{dim_label} — Health-score: full={full_h.mean():.1f}% (coverage={cov_h.mean()*100:.0f}%, "
              f"avg grid-point support={frac_h.mean()*100:.0f}%), restricted={restr_h.mean():.1f}%")
 
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"Action-sensitivity: full sweep vs. {title_suffix}", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved to {save_path}")
    return fig, results
 

 
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







if __name__ == "__main__":
    main()