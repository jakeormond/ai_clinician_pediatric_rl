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

import torch

from src.rl.models.bc_with_logprobs import Actor


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
    config_manager = None
    if os.path.isfile(config_manager_path):
        with open(config_manager_path, "rb") as f:
            config_manager = pickle.load(f)
        cm_reward_type = getattr(config_manager, "reward_type", None)
        if cm_reward_type is not None and cm_reward_type != reward_type:
            print(f"Warning: requested reward_type='{reward_type}' but "
                  f"config_manager.reward_type='{cm_reward_type}' for {model_dir}")
    else:
        print(f"Warning: no config_manager.pkl found at {config_manager_path} -- returning None for config_manager.")

    actor = Actor(
        state_dim=state_dim,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams.get("dropout", 0.1),
    ).to(device)

    final_model_path = os.path.join(model_dir, "final_model.pt")
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)

    # BC's checkpoint key naming has some uncertainty (Q_Learner/CQL's
    # checkpoints use "actor_state_dict"; BC.state_dict() itself returns
    # a dict keyed "actor" -- depending on which path your training.py
    # checkpoint-saving branch takes for 'bc_with_logprobs', either could
    # be correct). Check both explicitly rather than guessing silently.
    if "actor_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["actor_state_dict"])
    elif "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
    else:
        raise KeyError(
            f"Could not find actor weights in checkpoint at {final_model_path}. "
            f"Checked keys 'actor_state_dict' and 'actor'; available keys were: "
            f"{list(checkpoint.keys())}. Update load_actor() with the correct key."
        )

    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    return actor, config_manager, hyperparams