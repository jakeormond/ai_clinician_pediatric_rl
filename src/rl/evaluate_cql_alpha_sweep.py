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

from .evaluate_terminal_reward_models import generate_model_outputs


from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..setup_environment import load_paths

ALPHA_COLOR = "#2471a3"


def get_model_dirs(base_dir):
    model_dirs = []
    # find directories in base_dir that begin with choc
    choc_dirs = [d for d in os.listdir(base_dir) if d.startswith("choc")]

    if choc_dirs == []:
        print(f"No choc directories found in {base_dir}")
        # look for directories that begin with unknown or local
        sub_dirs = [d for d in os.listdir(base_dir) if d.startswith("unknown") or d.startswith("local")]
        for sub_dir in sub_dirs:
            model_dirs.append(os.path.join(base_dir, sub_dir))
            print(f"Found model directory: {os.path.join(base_dir, sub_dir)}")
    else:
        for choc_dir in choc_dirs:
            # find directories that begin with unknown or local
            sub_dirs = [d for d in os.listdir(os.path.join(base_dir, choc_dir)) if d.startswith("unknown") or d.startswith("local")]
            for sub_dir in sub_dirs:
                model_dirs.append(os.path.join(base_dir, choc_dir, sub_dir))
                print(f"Found model directory: {os.path.join(base_dir, choc_dir, sub_dir)}")
    return model_dirs      


"""
Stage 2 CQL alpha sweep analysis, run on the VAL set only (alpha selection
is a modeling decision — test set stays reserved for the final Stage 3
5-seed evaluation).
 
For each alpha's model, computes:
  1. Own-critic advantage: Q_own(actor_action) - Q_own(clinician_action)
     -- an exploitation/red-flag check, not positive evidence on its own.
  2. SARSA-critic advantage, evaluated against ALL 5 SARSA seeds separately:
     Q_sarsa_i(actor_action) - Q_sarsa_i(clinician_action) for i in 1..5
     -- the primary evidence: independent, unregularized, never-adversarial
     critics corroborating (or not) that the actor found real improvement.
     Reported as mean + spread across the 5 SARSA seeds per alpha, so you
     can see whether an apparent improvement is robust to which SARSA seed
     judges it.
  3. Occupancy support of the actor's ACTUAL chosen actions (not a sweep
     grid) — what fraction of states get a recommendation whose (severity,
     fluid, vaso) combination was well-supported in training data.
 
Expects, already in your session:
    generate_model_outputs                          (your existing function)
    build_joint_state_action_occupancy_counts, _bin_index   (occupancy module)
    processors['val']                                 (VAL processor — see note above)
    sarsa_models  : list of 5 loaded SARSA critics (one per seed), e.g.
                    [(critic, config, hyperparams), ...] or bare critics —
                    handled either way below
    training_actions, training_health_score           (for the occupancy histogram)
"""


 
def get_actor_actions(model_entry, processor, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    return generate_model_outputs(
        processor, model=actor, model_type="actor", batch_size=batch_size, device=device
    )
 
 
def get_critic(model_entry):
    """Handles (actor, critic, ...) CQL tuples and (critic, ...) SARSA tuples/bare critics."""
    if isinstance(model_entry, (list, tuple)):
        # CQL tuples are (actor, critic, config, hyperparams); SARSA tuples are (critic, config, hyperparams)
        # heuristic: an nn.Module with a `forward` taking (state, action) and returning 2 outputs is a critic —
        # simplest robust check here is just "the second element if length>=2 and first looks like an actor,
        # else the first element". Safer to be explicit if you know your own tuple shapes:
        return model_entry[1] if len(model_entry) >= 2 and hasattr(model_entry[0], "gate_temperature") else model_entry[0]
    return model_entry
 
 
def compute_model_advantage(
        model_entry, processor, sarsa_models,
        batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    Returns a dict: own_advantage (per-timestep array), sarsa_advantages
    (list of 5 per-timestep arrays, one per SARSA seed), actor_actions.
    """
    own_critic = get_critic(model_entry)
 
    actor_actions = get_actor_actions(model_entry, processor, batch_size=batch_size, device=device)
 
    q_own_clinician = generate_model_outputs(
        processor, model=own_critic, actions_to_use=None, model_type="critic",
        reduction='mean', batch_size=batch_size, device=device,
    )
    q_own_actor = generate_model_outputs(
        processor, model=own_critic, actions_to_use=actor_actions, model_type="critic",
        reduction='mean', batch_size=batch_size, device=device,
    )
    own_advantage = (q_own_actor - q_own_clinician).cpu().numpy()
 
    sarsa_advantages = []
    for seed_idx, sarsa_entry in enumerate(sarsa_models):
        sarsa_critic = get_critic(sarsa_entry)
        q_sarsa_clinician = generate_model_outputs(
            processor, model=sarsa_critic, actions_to_use=None, model_type="critic",
            reduction='mean', batch_size=batch_size, device=device,
        )
        q_sarsa_actor = generate_model_outputs(
            processor, model=sarsa_critic, actions_to_use=actor_actions, model_type="critic",
            reduction='mean', batch_size=batch_size, device=device,
        )
        sarsa_advantages.append((q_sarsa_actor - q_sarsa_clinician).cpu().numpy())
 
    return {
        "own_advantage": own_advantage,
        "sarsa_advantages": sarsa_advantages,  # list of 5 arrays
        "actor_actions": actor_actions,
    }
 
 
def compute_occupancy_support_of_actions(
        actions, test_health_score, training_actions, training_health_score,
        n_severity_bins=20, n_action_bins=10, min_count=10,
    ):
    """
    Fraction of timesteps where the actor's ACTUAL chosen (fluid, vaso)
    combination, given that timestep's own severity bin, was well-supported
    (count >= min_count) in the training data.
    """
    joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
    )
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    hs_np = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
 
    sev_bin = _bin_index(hs_np, severity_edges)
    fluid_bin = _bin_index(actions_np[:, 0], fluid_edges)
    vaso_bin = _bin_index(actions_np[:, 1], vaso_edges)
    counts = joint_counts[sev_bin, fluid_bin, vaso_bin]
 
    return (counts >= min_count).mean(), counts
 
 
def run_alpha_sweep_analysis(
        cql_models_by_alpha,   # dict: {alpha_value: model_entry}
        processor, sarsa_models,
        val_health_score, training_actions, training_health_score,
        n_severity_bins=20, n_action_bins=10, min_count=10,
        save_path=None,
    ):
    alphas = sorted(cql_models_by_alpha.keys())
    results = {}
 
    for alpha in alphas:
        t0 = time.time()
        print(f"=== alpha={alpha} ===", flush=True)
        model_entry = cql_models_by_alpha[alpha]
 
        adv = compute_model_advantage(model_entry, processor, sarsa_models)
        occ_frac, _ = compute_occupancy_support_of_actions(
            adv["actor_actions"], val_health_score, training_actions, training_health_score,
            n_severity_bins=n_severity_bins, n_action_bins=n_action_bins, min_count=min_count,
        )
 
        sarsa_means = [a.mean() for a in adv["sarsa_advantages"]]
        results[alpha] = {
            "own_advantage_mean": adv["own_advantage"].mean(),
            "sarsa_advantage_means": np.array(sarsa_means),  # 5 values, one per SARSA seed
            "occupancy_support_frac": occ_frac,
        }
 
        print(f"  own-critic advantage:   {adv['own_advantage'].mean():.4f}")
        print(f"  SARSA-critic advantage: mean={np.mean(sarsa_means):.4f}, "
              f"per-seed={np.round(sarsa_means, 4)}")
        print(f"  occupancy support of chosen actions: {occ_frac*100:.1f}%")
        print(f"  done in {time.time()-t0:.1f}s\n", flush=True)
 
    # --- plot: 3 panels vs alpha ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
 
    own_vals = [results[a]["own_advantage_mean"] for a in alphas]
    axes[0].plot(alphas, own_vals, marker="o", color="#c0392b")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("cql_alpha")
    axes[0].set_ylabel("Own-critic advantage")
    axes[0].set_title("Own-critic advantage\n(exploitation red-flag, not positive evidence)", fontsize=10)
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].spines[["top", "right"]].set_visible(False)
 
    sarsa_mean_vals = [results[a]["sarsa_advantage_means"].mean() for a in alphas]
    sarsa_min_vals = [results[a]["sarsa_advantage_means"].min() for a in alphas]
    sarsa_max_vals = [results[a]["sarsa_advantage_means"].max() for a in alphas]
    axes[1].plot(alphas, sarsa_mean_vals, marker="o", color=ALPHA_COLOR, label="mean across 5 SARSA seeds")
    axes[1].fill_between(alphas, sarsa_min_vals, sarsa_max_vals, color=ALPHA_COLOR, alpha=0.2, label="min-max across seeds")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("cql_alpha")
    axes[1].set_ylabel("SARSA-critic advantage")
    axes[1].set_title("SARSA-critic advantage\n(primary evidence of real improvement)", fontsize=10)
    axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
 
    occ_vals = [results[a]["occupancy_support_frac"] * 100 for a in alphas]
    axes[2].plot(alphas, occ_vals, marker="o", color="#229954")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("cql_alpha")
    axes[2].set_ylabel("% of actions well-supported")
    axes[2].set_title("Occupancy support of\nactor's chosen actions", fontsize=10)
    axes[2].spines[["top", "right"]].set_visible(False)
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
 
    return results, fig
 