"""
Analysis support for the 5 final BC ("behavioral cloning") seed models --
mirrors the structure of evaluate_cql_alpha_sweep.py / evaluate_terminal_reward_models.py.

BC's trained actor is used downstream as the behavior-policy density
estimate for weighted importance sampling (WIS) -- see log_prob_for_wis
in cql_with_logprobs.py, which borrows BC's fitted magnitude std as its
bandwidth. Getting this loader (and, eventually, BC's own std-floor/
mean-saturation diagnostics) right matters directly for WIS quality.
"""

import os
import json
import pickle
import numpy as np

import torch

from .models.bc_with_logprobs import Actor


def load_actor(reward_type, model_dir, state_dim, device=None):
    """
    Loads one trained BC actor (+ its config_manager and hyperparameters)
    from a specific, already-resolved run directory.

    reward_type: expected reward type for this model (e.g. 'health_score')
        -- used only as a SANITY CHECK against the loaded config_manager's
        own .reward_type, not to resolve any path (model_dir is already
        fully resolved by the caller). A mismatch prints a warning rather
        than raising, since this is meant to catch accidental mix-ups
        without being overly strict about config_manager's exact attribute
        name/presence.

    model_dir: path to one specific run's directory (containing
        results.json, config_manager.pkl, and final_model.pt).

    state_dim: passed in directly (e.g. from Dynamic DeepHit's d_model)
        since BC's Actor, like CQL's, does not store this itself.

    Returns: (actor, config_manager, hyperparams)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_path = os.path.join(model_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    hyperparams = results["hyperparameters"]

    config_manager_path = os.path.join(model_dir, "config_manager.pkl")
    if not os.path.isfile(config_manager_path):
        # sometimes config_manager.pkl sits in the parent directory instead
        # (e.g. shared across a run's seed subfolders) -- check there before
        # giving up.
        parent_config_manager_path = os.path.join(os.path.dirname(model_dir), "config_manager.pkl")
        if os.path.isfile(parent_config_manager_path):
            config_manager_path = parent_config_manager_path
        else:
            raise FileNotFoundError(
                f"No config_manager.pkl found at {os.path.join(model_dir, 'config_manager.pkl')} "
                f"or in the parent directory ({parent_config_manager_path})."
            )

    with open(config_manager_path, "rb") as f:
        config_manager = pickle.load(f)
    cm_reward_type = getattr(config_manager, "reward_type", None)
    if cm_reward_type is not None and cm_reward_type != reward_type:
        print(f"Warning: requested reward_type='{reward_type}' but "
              f"config_manager.reward_type='{cm_reward_type}' for {model_dir}")

    final_model_path = os.path.join(model_dir, "final_model.pt")
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)

    # BC's checkpoint key naming has some uncertainty (Q_Learner/CQL's
    # checkpoints use "actor_state_dict"; BC.state_dict() itself returns
    # a dict keyed "actor" -- depending on which path your training.py
    # checkpoint-saving branch takes for 'bc_with_logprobs', either could
    # be correct). Check both explicitly rather than guessing silently.
    if "actor_state_dict" in checkpoint:
        actor_state_dict = checkpoint["actor_state_dict"]
    elif "actor" in checkpoint:
        actor_state_dict = checkpoint["actor"]
    else:
        raise KeyError(
            f"Could not find actor weights in checkpoint at {final_model_path}. "
            f"Checked keys 'actor_state_dict' and 'actor'; available keys were: "
            f"{list(checkpoint.keys())}. Update load_actor() with the correct key."
        )

    # Infer fluid_gate_hidden DIRECTLY from the checkpoint's own weight shapes,
    # rather than trusting hyperparams/results.json to have recorded it --
    # results.json may not log every hyperparameter key (confirmed: some
    # fluid_gate_hidden runs' results.json omitted it entirely, causing a
    # silent fallback to 0 and a state_dict key mismatch). Reading the
    # checkpoint's actual structure can't be wrong, since it IS the
    # structure that was actually saved.
    if "fluid_gate.0.weight" in actor_state_dict:
        fluid_gate_hidden = actor_state_dict["fluid_gate.0.weight"].shape[0]
    elif "fluid_gate.weight" in actor_state_dict:
        fluid_gate_hidden = 0
    else:
        raise KeyError(
            f"Could not find fluid_gate weights in checkpoint at {final_model_path} "
            f"under either 'fluid_gate.weight' (plain Linear) or 'fluid_gate.0.weight' "
            f"(Sequential MLP). Available keys: {list(actor_state_dict.keys())}"
        )

    actor = Actor(
        state_dim=state_dim,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams.get("dropout", 0.1),
        fluid_gate_hidden=fluid_gate_hidden,
    ).to(device)

    load_result = actor.load_state_dict(actor_state_dict, strict=False)
    # strict=False tolerates OLDER checkpoints (e.g. saved before the
    # fluid_gate_calib_scale/bias buffers existed) missing those two keys --
    # they simply keep their default, no-op (identity) values in that case.
    # Anything else missing/unexpected would indicate a real mismatch, so
    # we still check and warn explicitly rather than silencing everything.
    unexpected_missing = [k for k in load_result.missing_keys if k not in
                          ("fluid_gate_calib_scale", "fluid_gate_calib_bias")]
    if unexpected_missing or load_result.unexpected_keys:
        print(f"Warning: unexpected state_dict mismatch loading {model_dir}: "
              f"missing={unexpected_missing}, unexpected={load_result.unexpected_keys}")

    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    return actor, config_manager, hyperparams


def get_bc_actor_actions(model_entry, processor, batch_size=1024,
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    BC-compatible drop-in replacement for action_distribution_sweep.get_actor_actions
    (which calls the CQL-style actor(state, deterministic=True) signature --
    BC's Actor.forward has no `deterministic` kwarg at all, since it only
    ever computes the deterministic path internally). Pass this function as
    `get_actor_actions_fn` to run_action_distribution_stage3 /
    plot_dose_histograms / plot_dose_histograms_at_clinician_zero.
    """
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    actor.eval()

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]

    n_total = processor.rl_dataset["states"].shape[0]
    action_dim = processor.rl_dataset["actions"].shape[1]
    actions_all = torch.zeros(n_total, action_dim, dtype=torch.float32).to(device)

    for stay_id in stay_ids:
        stay_idx = np.where(stay_ids == stay_id)[0][0]
        start, end = traj_start_inds[stay_idx], traj_start_inds[stay_idx + 1]
        states = torch.as_tensor(processor.rl_dataset["states"][start:end], dtype=torch.float32).to(device)

        n_batches = int(np.ceil(states.shape[0] / batch_size))
        for i in range(n_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, states.shape[0])
            states_batch = states[s:e]
            with torch.no_grad():
                actions, _, _ = actor(states_batch)
            actions_all[start + s:start + e] = actions

    return actions_all


def get_gate_probs(model_entry, processor, batch_size=1024,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Returns (f_p, v_p) as numpy arrays, shape (n_timesteps,) -- the raw
    fluid/vaso gate firing PROBABILITIES (pre-threshold), for every state
    in the processor's dataset. Same trajectory-batching pattern as
    check_std_floor.get_both_std, but extracting f_p/v_p instead of std.
    """
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    actor.eval()

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]

    n_total = processor.rl_dataset["states"].shape[0]
    f_p_all = torch.zeros(n_total, dtype=torch.float32)
    v_p_all = torch.zeros(n_total, dtype=torch.float32)

    for stay_id in stay_ids:
        stay_idx = np.where(stay_ids == stay_id)[0][0]
        start, end = traj_start_inds[stay_idx], traj_start_inds[stay_idx + 1]
        states = torch.as_tensor(processor.rl_dataset["states"][start:end], dtype=torch.float32).to(device)

        n_batches = int(np.ceil(states.shape[0] / batch_size))
        for i in range(n_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, states.shape[0])
            states_batch = states[s:e]
            with torch.no_grad():
                _, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = actor(states_batch, return_dist=True)
            f_p_all[start + s:start + e] = f_p.squeeze(-1).cpu()
            v_p_all[start + s:start + e] = v_p.squeeze(-1).cpu()

    return f_p_all.numpy(), v_p_all.numpy()


def check_gate_calibration_by_seed(models_by_seed, processor, clinician_actions, zero_threshold=1e-4):
    """
    For each seed, checks whether the raw gate probability (f_p, v_p)
    genuinely distinguishes clinician-zero from clinician-nonzero states,
    or whether it's uniformly high (a real calibration failure) regardless
    of what the clinician actually did.
    """
    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    clinician_fluid_fired = clinician_actions_np[:, 0] > zero_threshold
    clinician_vaso_fired = clinician_actions_np[:, 1] > zero_threshold

    for seed_idx, model_entry in enumerate(models_by_seed):
        f_p, v_p = get_gate_probs(model_entry, processor)

        print(f"=== Seed {seed_idx} ===")
        for name, p, fired_mask in [("Fluid", f_p, clinician_fluid_fired), ("Vaso", v_p, clinician_vaso_fired)]:
            p_at_zero = p[~fired_mask]
            p_at_nonzero = p[fired_mask]
            pct_over_half_at_zero = (p_at_zero > 0.5).mean() * 100
            print(f"  {name} gate probability (f_p):")
            print(f"    At clinician-ZERO states:    mean={p_at_zero.mean():.3f}, median={np.median(p_at_zero):.3f}, "
                  f"p10={np.percentile(p_at_zero,10):.3f}, p90={np.percentile(p_at_zero,90):.3f}, "
                  f"%>0.5={pct_over_half_at_zero:.1f}%")
            print(f"    At clinician-NONZERO states: mean={p_at_nonzero.mean():.3f}, median={np.median(p_at_nonzero):.3f}, "
                  f"p10={np.percentile(p_at_nonzero,10):.3f}, p90={np.percentile(p_at_nonzero,90):.3f}")
        print()


def check_fluid_gate_wis_impact(models_by_seed, processor, clinician_actions, zero_threshold=1e-4):
    """
    Quantifies how much the fluid gate's miscalibration at clinician-zero
    states could distort WIS importance weights, WITHOUT needing the
    target (CQL) policy -- two complementary views:

      1. Direct probability mass: what probability does BC actually assign
         to the real "no fluid" decision at these states (1 - f_p), vs.
         the analogous, well-calibrated vaso number (1 - v_p) at
         clinician-zero-vaso states, as a "what good looks like" reference.

      2. Trajectory-level compounding: for each patient trajectory, what
         FRACTION of the total gate-related negative-log-likelihood (i.e.
         "surprise") comes specifically from zero-fluid states, vs.
         everything else (nonzero-fluid gate decisions + all vaso gate
         decisions). A large fraction here means this single, known
         calibration issue could dominate that trajectory's resulting
         importance weight; a small fraction means it's a minor
         correction relative to everything else already being modeled.
    """
    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    clinician_fluid_fired = clinician_actions_np[:, 0] > zero_threshold
    clinician_vaso_fired = clinician_actions_np[:, 1] > zero_threshold

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    n_traj = len(traj_end_inds)

    for seed_idx, model_entry in enumerate(models_by_seed):
        f_p, v_p = get_gate_probs(model_entry, processor)
        eps = 1e-8

        # --- 1. direct probability mass at clinician-zero states ---
        prob_correct_fluid_zero = 1 - f_p[~clinician_fluid_fired]
        prob_correct_vaso_zero = 1 - v_p[~clinician_vaso_fired]

        print(f"=== Seed {seed_idx} ===")
        print(f"  Probability BC assigns to the REAL 'no fluid' decision, at clinician-zero-fluid states:")
        print(f"    median={np.median(prob_correct_fluid_zero):.3f}, mean={prob_correct_fluid_zero.mean():.3f}")
        print(f"  Probability BC assigns to the REAL 'no vaso' decision, at clinician-zero-vaso states (reference -- well calibrated):")
        print(f"    median={np.median(prob_correct_vaso_zero):.3f}, mean={prob_correct_vaso_zero.mean():.3f}")

        # --- 2. trajectory-level compounding ---
        f_gate_lp_all = np.log(np.where(clinician_fluid_fired, f_p, 1 - f_p) + eps)
        v_gate_lp_all = np.log(np.where(clinician_vaso_fired, v_p, 1 - v_p) + eps)

        zero_fluid_mask = ~clinician_fluid_fired

        fractions = []
        for t in range(n_traj):
            start, end = traj_start_inds[t], traj_start_inds[t + 1]
            f_gate_lp_traj = f_gate_lp_all[start:end]
            v_gate_lp_traj = v_gate_lp_all[start:end]
            zero_fluid_traj = zero_fluid_mask[start:end]

            total_gate_nll = -(f_gate_lp_traj.sum() + v_gate_lp_traj.sum())
            zero_fluid_nll = -f_gate_lp_traj[zero_fluid_traj].sum()

            if total_gate_nll > eps:
                fractions.append(zero_fluid_nll / total_gate_nll)

        fractions = np.array(fractions)
        print(f"  Fraction of each trajectory's TOTAL gate log-likelihood magnitude coming from "
              f"zero-fluid states specifically (n={n_traj} trajectories):")
        print(f"    median={np.median(fractions)*100:.1f}%, mean={fractions.mean()*100:.1f}%, "
              f"p90={np.percentile(fractions,90)*100:.1f}%")
        print()