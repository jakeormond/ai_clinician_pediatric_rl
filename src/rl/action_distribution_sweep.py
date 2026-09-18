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
        clinician_baseline=None,
        zero_threshold=1e-4, batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        title="", save_path=None,
    ):
    """
    Runs compute_action_distribution_stats for every alpha in
    cql_models_by_alpha, prints a per-alpha summary, and plots fire rate /
    mean dose / mean non-zero dose vs. alpha, as 3 panels with fluid and
    vaso overlaid in each (color-coded), plus a color-matched dashed
    clinician reference line per drug.

    clinician_baseline: optional dict from compute_clinician_baseline(),
        e.g. {"fluid_fire_rate": ..., "fluid_mean_dose": ..., ...}.
        If given, drawn as a horizontal reference line, color-matched to
        each drug, on each panel.
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panel_specs = [
        ("fire_rate", "Firing rate (%)", 100.0),
        ("mean_dose", "Mean dose (all timesteps)", 1.0),
        ("mean_nonzero_dose", "Mean dose (fired timesteps only)", 1.0),
    ]

    for col, (stat_suffix, ylabel, mult) in enumerate(panel_specs):
        ax = axes[col]
        for name, color in [("fluid", FLUID_COLOR), ("vaso", VASO_COLOR)]:
            key = f"{name}_{stat_suffix}"
            vals = [results[a][key] * mult for a in alphas]
            ax.plot(alphas, vals, marker="o", color=color, label=f"{name} (CQL actor)")

            if clinician_baseline is not None and key in clinician_baseline:
                baseline_val = clinician_baseline[key] * mult
                ax.axhline(baseline_val, color=color, linewidth=1.2, linestyle="--", alpha=0.6,
                           label=f"{name} (clinician)")

        ax.set_xlabel("cql_alpha")
        ax.set_title(ylabel, fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title or "Actor action distribution across alpha sweep", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return results, fig


def run_action_distribution_stage3(
        models_by_seed,   # list of 5 model_entry tuples/objects, one per seed, at the FIXED final alpha
        processor,
        clinician_baseline=None,
        zero_threshold=1e-4, batch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
        get_actor_actions_fn=None,   # optional override, e.g. eval_bc.get_bc_actor_actions for BC actors
        title="", save_path=None,
    ):
    """
    Stage 3 variant: seeds (at a single, fixed final alpha) as the unit of
    replication, rather than sweeping alpha. Plots the same 3 stats
    (fire rate, mean dose, mean non-zero dose), fluid+vaso overlaid, but
    as a beeswarm of per-seed points rather than a connected line -- seeds
    have no natural order, so a line would misleadingly imply one.

    models_by_seed: list of 5 model_entry tuples/objects (e.g.
        cql_health_score_models filtered to the final alpha, or the Stage 3
        list directly if it only contains that one alpha).

    get_actor_actions_fn: optional function(model_entry, processor,
        batch_size=..., device=...) -> actions tensor, overriding the
        default action_dist.get_actor_actions (which assumes a CQL-style
        actor(state, deterministic=True) signature). Pass a compatible
        function for other actor types, e.g. BC's Actor.forward, which has
        no `deterministic` kwarg at all.
    """
    action_fn = get_actor_actions_fn if get_actor_actions_fn is not None else get_actor_actions

    n_seeds = len(models_by_seed)
    per_seed_stats = []

    for seed_idx, model_entry in enumerate(models_by_seed):
        actions = action_fn(model_entry, processor, batch_size=batch_size, device=device)
        stats = compute_action_distribution_stats(actions, zero_threshold=zero_threshold)
        per_seed_stats.append(stats)

        print(f"seed {seed_idx+1}/{n_seeds}: "
              f"fluid[rate={stats['fluid_fire_rate']*100:.1f}%, mean={stats['fluid_mean_dose']:.4f}, "
              f"nonzero_mean={stats['fluid_mean_nonzero_dose']:.4f}]  "
              f"vaso[rate={stats['vaso_fire_rate']*100:.1f}%, mean={stats['vaso_mean_dose']:.4f}, "
              f"nonzero_mean={stats['vaso_mean_nonzero_dose']:.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panel_specs = [
        ("fire_rate", "Firing rate (%)", 100.0),
        ("mean_dose", "Mean dose (all timesteps)", 1.0),
        ("mean_nonzero_dose", "Mean dose (fired timesteps only)", 1.0),
    ]

    rng = np.random.default_rng(0)  # fixed seed, reproducible jitter

    for col, (stat_suffix, ylabel, mult) in enumerate(panel_specs):
        ax = axes[col]
        for x_center, (name, color) in enumerate([("fluid", FLUID_COLOR), ("vaso", VASO_COLOR)]):
            key = f"{name}_{stat_suffix}"
            vals = np.array([s[key] * mult for s in per_seed_stats])
            jitter = rng.uniform(-0.08, 0.08, size=n_seeds)
            ax.scatter(x_center + jitter, vals, color=color, s=60, alpha=0.85,
                       edgecolors="white", linewidths=0.7, zorder=3,
                       label=f"{name} (CQL actor, per seed)")

            if clinician_baseline is not None and key in clinician_baseline:
                baseline_val = clinician_baseline[key] * mult
                ax.hlines(baseline_val, x_center - 0.25, x_center + 0.25, color=color,
                          linewidth=1.5, linestyle="--", alpha=0.7,
                          label=f"{name} (clinician)")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Fluid", "Vaso"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(ylabel, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="best")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title or f"Actor action distribution across {n_seeds} seeds (final alpha)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return per_seed_stats, fig


if __name__ == "__main__":
    from src.rl.clinician_baseline import compute_clinician_baseline

    clinician_baseline_test = compute_clinician_baseline(processors['test'].rl_dataset["actions"])

    stats_health_score, fig_health_score = run_action_distribution_stage3(
        cql_health_score_models,  # the 5 final Stage 3 models at alpha=0.006
        processors['test'],
        clinician_baseline=clinician_baseline_test,
        title="Health-score reward: final policy action distribution (5 seeds, test set)",
        save_path="stage3_actions_health_score.png",
    )

    stats_terminal, fig_terminal = run_action_distribution_stage3(
        cql_terminal_models,  # the 5 final Stage 3 models at terminal's final alpha
        processors['test'],
        clinician_baseline=clinician_baseline_test,
        title="Terminal reward: final policy action distribution (5 seeds, test set)",
        save_path="stage3_actions_terminal.png",
    )