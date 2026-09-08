"""
Diagnostic: actor action distribution (fluid, vaso) across the CQL alpha
sweep. Complements the advantage/occupancy analysis by directly showing
WHAT the actor is recommending at each alpha, rather than just how good
those recommendations look to a critic.

For each alpha's model, computes, for both fluid and vaso:
    - fire rate (fraction of timesteps with dose > threshold)
    - mean dose (across all timesteps, zeros included)
    - mean non-zero dose (across only timesteps where the drug fired)

Expects, already in your session (or importable):
    generate_model_outputs
    processors['val']
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.rl.evaluate_terminal_reward_models import generate_model_outputs

FLUID_COLOR = "#2471a3"
VASO_COLOR = "#c0392b"


def get_actor_actions(model_entry, processor, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    return generate_model_outputs(
        processor, model=actor, model_type="actor", batch_size=batch_size, device=device
    )


def compute_action_distribution_stats(actions, zero_threshold=1e-4):
    """
    actions: (n_timesteps, 2) tensor/array, columns = [fluid, vaso]
    Returns a dict of stats for each dimension.
    """
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    stats = {}
    for dim, name in [(0, "fluid"), (1, "vaso")]:
        dose = actions_np[:, dim]
        fired = dose > zero_threshold
        stats[f"{name}_fire_rate"] = fired.mean()
        stats[f"{name}_mean_dose"] = dose.mean()
        stats[f"{name}_mean_nonzero_dose"] = dose[fired].mean() if fired.any() else np.nan
    return stats


def run_action_distribution_sweep(
        cql_models_by_alpha, processor,
        zero_threshold=1e-4, batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        title="", save_path=None,
    ):
    """
    Runs compute_action_distribution_stats for every alpha in
    cql_models_by_alpha, prints a per-alpha summary, and plots fire rate /
    mean dose / mean non-zero dose vs. alpha for both fluid and vaso.
    """
    alphas = sorted(cql_models_by_alpha.keys())
    results = {}

    for alpha in alphas:
        model_entry = cql_models_by_alpha[alpha]
        actions = get_actor_actions(model_entry, processor, batch_size=batch_size, device=device)
        stats = compute_action_distribution_stats(actions, zero_threshold=zero_threshold)
        results[alpha] = stats

        print(f"alpha={alpha}: "
              f"fluid[rate={stats['fluid_fire_rate']*100:.1f}%, mean={stats['fluid_mean_dose']:.4f}, "
              f"nonzero_mean={stats['fluid_mean_nonzero_dose']:.4f}]  "
              f"vaso[rate={stats['vaso_fire_rate']*100:.1f}%, mean={stats['vaso_mean_dose']:.4f}, "
              f"nonzero_mean={stats['vaso_mean_nonzero_dose']:.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for name, color in [("fluid", FLUID_COLOR), ("vaso", VASO_COLOR)]:
        rate_vals = [results[a][f"{name}_fire_rate"] * 100 for a in alphas]
        mean_vals = [results[a][f"{name}_mean_dose"] for a in alphas]
        nonzero_vals = [results[a][f"{name}_mean_nonzero_dose"] for a in alphas]

        axes[0].plot(alphas, rate_vals, marker="o", color=color, label=name)
        axes[1].plot(alphas, mean_vals, marker="o", color=color, label=name)
        axes[2].plot(alphas, nonzero_vals, marker="o", color=color, label=name)

    axes[0].set_xlabel("cql_alpha")
    axes[0].set_ylabel("Fire rate (%)")
    axes[0].set_title("Firing rate", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].set_xlabel("cql_alpha")
    axes[1].set_ylabel("Mean dose (all timesteps)")
    axes[1].set_title("Mean dose", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].set_xlabel("cql_alpha")
    axes[2].set_ylabel("Mean dose (fired timesteps only)")
    axes[2].set_title("Mean non-zero dose", fontsize=10)
    axes[2].spines[["top", "right"]].set_visible(False)

    fig.suptitle(title or "Actor action distribution across alpha sweep", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return results, fig


if __name__ == "__main__":
    results_actions_terminal, fig_actions_terminal = run_action_distribution_sweep(
        cql_terminal_by_alpha,
        processors['val'],
        title="Terminal reward: actor action distribution vs. alpha",
        save_path="cql_alpha_sweep_actions_terminal.png",
    )