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

from kneed import KneeLocator

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

# from .evaluate_terminal_reward_models import generate_model_outputs
from src.rl.evaluate_terminal_reward_models import generate_model_outputs, \
    build_joint_state_action_occupancy_counts, _bin_index, action_sensitivity_sweep


from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..setup_environment import load_paths

ALPHA_COLOR = "#2471a3"

OWN_COLOR = "#c0392b"
SARSA_COLOR = "#2471a3"
OCC_COLOR = "#229954"
KNEE_COLOR = "#7d3c98"


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
 

def _iqr_normalize(values, reference_q):
    """
    Normalizes `values` (e.g. an advantage array) by the interquartile range
    of `reference_q` (that critic's own Q-values under real clinician
    actions) -- same principle as normalize_sensitivity elsewhere in this
    pipeline, so advantage is comparable across critics with very different
    absolute Q scales (e.g. terminal vs. health-score).
    """
    q25, q75 = torch.quantile(reference_q, 0.25), torch.quantile(reference_q, 0.75)
    iqr = (q75 - q25).clamp(min=1e-6)
    return (values / iqr)


def compute_model_advantage(
        model_entry, processor, sarsa_models,
        batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    Returns a dict with both RAW and IQR-NORMALIZED advantage:
      own_advantage / own_advantage_normalized       (per-timestep arrays)
      sarsa_advantages / sarsa_advantages_normalized  (lists of 5 per-timestep arrays)
      actor_actions
 
    Normalization divides each critic's advantage by the interquartile
    range of THAT SAME critic's own Q-values under real clinician actions
    -- so own-critic advantage is normalized by the own critic's IQR, and
    each SARSA critic's advantage is normalized by that SAME SARSA critic's
    own IQR (not shared across seeds or across own/SARSA). This puts
    advantage on a comparable, scale-free footing across critics/reward
    types that otherwise sit on very different absolute Q scales (e.g.
    terminal's MC-return-based Q vs. health-score's summed-difference Q).
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
    own_advantage_raw = q_own_actor - q_own_clinician
    own_advantage_normalized = _iqr_normalize(own_advantage_raw, q_own_clinician)
 
    sarsa_advantages_raw = []
    sarsa_advantages_normalized = []
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
        sarsa_adv_raw = q_sarsa_actor - q_sarsa_clinician
        sarsa_advantages_raw.append(sarsa_adv_raw.cpu().numpy())
        sarsa_advantages_normalized.append(_iqr_normalize(sarsa_adv_raw, q_sarsa_clinician).cpu().numpy())
 
    return {
        "own_advantage": own_advantage_raw.cpu().numpy(),
        "own_advantage_normalized": own_advantage_normalized.cpu().numpy(),
        "sarsa_advantages": sarsa_advantages_raw,
        "sarsa_advantages_normalized": sarsa_advantages_normalized,
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
        sarsa_means_normalized = [a.mean() for a in adv["sarsa_advantages_normalized"]]
        results[alpha] = {
            "own_advantage_mean": adv["own_advantage"].mean(),
            "own_advantage_mean_normalized": adv["own_advantage_normalized"].mean(),
            "sarsa_advantage_means": np.array(sarsa_means),
            "sarsa_advantage_means_normalized": np.array(sarsa_means_normalized),
            "occupancy_support_frac": occ_frac,
        }
 
        print(f"  own-critic advantage:   raw={adv['own_advantage'].mean():.4f}, "
              f"normalized={adv['own_advantage_normalized'].mean():.4f}")
        print(f"  SARSA-critic advantage: raw mean={np.mean(sarsa_means):.4f}, "
              f"normalized mean={np.mean(sarsa_means_normalized):.4f} "
              f"(normalized per-seed={np.round(sarsa_means_normalized, 4)})")
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
 
 
def find_knee_alpha(results, metric="sarsa_advantage", curve="convex", direction="decreasing", S=1.0,
                     alpha_min=None, normalized=False):
    """
    Identifies the knee/elbow of a chosen metric vs. alpha, directly from
    the results dict (no need to hand-copy values out).
 
    metric: which curve to run Kneedle on --
        "sarsa_advantage" -- mean SARSA-critic advantage (default)
        "own_advantage"   -- own-critic advantage
        "occupancy"       -- occupancy support (%) -- this one INCREASES,
                              so direction should be "increasing" if used
    normalized: if True, uses the IQR-normalized advantage values
        (own_advantage_mean_normalized / sarsa_advantage_means_normalized)
        instead of raw Q-unit advantage. Has no effect on "occupancy",
        which is never normalized (it's already a %). Use normalized=True
        when comparing knees across critics/reward types on different
        absolute Q scales (e.g. terminal vs. health-score).
    curve/direction: passed straight to KneeLocator -- see its docs for
        the four curve/direction combinations (convex/concave x
        increasing/decreasing). Must match the actual shape of the chosen
        metric or the result will be meaningless.
    alpha_min: if set, restricts the fit to alphas >= alpha_min -- use this
        when the full curve isn't monotonic in the expected direction
        (e.g., an early noisy dip before the real increasing trend starts).
    S: KneeLocator's sensitivity parameter -- higher S finds fewer, more
        conservative knees; lower S is more sensitive to smaller bends.
        Default (1.0) is KneeLocator's own default.
 
    Returns the alpha value at the identified knee (or None if Kneedle
    doesn't find one for this curve/direction/S combination).
    """
    alphas = sorted(results.keys())
    if alpha_min is not None:
        alphas = [a for a in alphas if a >= alpha_min]
 
    suffix = "_normalized" if normalized else ""
    if metric == "sarsa_advantage":
        y = [results[a][f"sarsa_advantage_means{suffix}"].mean() for a in alphas]
    elif metric == "own_advantage":
        y = [results[a][f"own_advantage_mean{suffix}"] for a in alphas]
    elif metric == "occupancy":
        y = [results[a]["occupancy_support_frac"] * 100 for a in alphas]
    else:
        raise ValueError(f"Unknown metric '{metric}'")
 
    kl = KneeLocator(alphas, y, curve=curve, direction=direction, S=S)
    return kl.elbow
 
 
def plot_combined_alpha_sweep(results, title="", save_path=None, mark_knee=True, normalized=False):
    """
    results: the dict returned by run_alpha_sweep_analysis, containing both
        raw and IQR-normalized advantage keys.
    normalized: if True, plots IQR-normalized advantage (comparable across
        critics/reward types on different absolute Q scales) instead of
        raw Q-unit advantage. Occupancy support is unaffected either way.
    """
    alphas = sorted(results.keys())
    suffix = "_normalized" if normalized else ""
 
    own_vals = np.array([results[a][f"own_advantage_mean{suffix}"] for a in alphas])
    sarsa_mean_vals = np.array([results[a][f"sarsa_advantage_means{suffix}"].mean() for a in alphas])
    sarsa_min_vals = np.array([results[a][f"sarsa_advantage_means{suffix}"].min() for a in alphas])
    sarsa_max_vals = np.array([results[a][f"sarsa_advantage_means{suffix}"].max() for a in alphas])
    occ_vals = np.array([results[a]["occupancy_support_frac"] * 100 for a in alphas])
 
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
 
    # left axis: both advantage curves, same units
    l1, = ax1.plot(alphas, own_vals, marker="o", color=OWN_COLOR, label="Own-critic advantage")
    l2, = ax1.plot(alphas, sarsa_mean_vals, marker="o", color=SARSA_COLOR, label="SARSA-critic advantage (mean)")
    ax1.fill_between(alphas, sarsa_min_vals, sarsa_max_vals, color=SARSA_COLOR, alpha=0.15,
                      label="SARSA min-max across 5 seeds")
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_xlabel(r"cql_$\alpha$")
    ax1.set_ylabel("Advantage (% of critic's own Q IQR)" if normalized else "Advantage (raw Q units)")
    ax1.spines[["top"]].set_visible(False)
 
    # right axis: occupancy support, different units (%)
    ax2 = ax1.twinx()
    l3, = ax2.plot(alphas, occ_vals, marker="s", color=OCC_COLOR, linestyle="--", label="Occupancy support (%)")
    ax2.set_ylabel("% of chosen actions well-supported", color=OCC_COLOR)
    ax2.tick_params(axis="y", colors=OCC_COLOR)
    ax2.spines["right"].set_color(OCC_COLOR)
    ax2.spines[["top"]].set_visible(False)
 
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
 
    if mark_knee:
        knee_alpha = find_knee_alpha(results, normalized=normalized)
        if knee_alpha is not None:
            l4 = ax1.axvline(knee_alpha, color=KNEE_COLOR, linewidth=1.5, linestyle=":",
                              label=f"Knee (Kneedle): $\\alpha$={knee_alpha}")
            lines.append(l4)
            labels.append(l4.get_label())
        else:
            print("Kneedle did not identify a knee for this curve.")
 
    ax1.legend(lines, labels, loc="upper right", frameon=False, fontsize=9)
    default_title = "CQL alpha sweep: advantage vs. occupancy support trade-off"
    ax1.set_title(title or (default_title + (" (normalized)" if normalized else "")), fontsize=12, fontweight="bold")
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
 
    return fig
