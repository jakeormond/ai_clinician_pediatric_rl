"""
Compares the actual distribution of chosen fluid (and vaso) doses at
several alpha values against the real clinician dose distribution --
diagnostic for the occupancy-support curve's shape (specifically: whether
a low/dipping occupancy value reflects a genuinely differentiated,
continuously-varying policy landing on many distinct dose values, vs. a
policy concentrated on a small number of common points).

Only plots NONZERO doses (i.e. conditional on firing) -- the fire
rate/mean-dose story is already covered by run_action_distribution_sweep;
this is specifically about the SHAPE of the dose distribution once a drug
is given.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

ALPHA_COLORS = ["#7d3c98", "#2471a3", "#229954", "#c0392b"]  # extend if using more than 4 alphas
CLINICIAN_COLOR = "#7f8c8d"


def get_nonzero_doses(actions, dim, zero_threshold=1e-4):
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    dose = actions_np[:, dim]
    return dose[dose > zero_threshold]


def _print_dose_summary(label, doses):
    if len(doses) == 0:
        print(f"  {label}: no nonzero doses")
        return
    p10, p25, p50, p75, p90 = np.percentile(doses, [10, 25, 50, 75, 90])
    print(f"  {label}: n={len(doses)}, mean={doses.mean():.4f}, std={doses.std():.4f}, "
          f"median={p50:.4f}, p10={p10:.4f}, p25={p25:.4f}, p75={p75:.4f}, p90={p90:.4f}")


def plot_dose_histograms(
        cql_models_by_alpha, alphas_to_show, processor, clinician_actions,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=None,  # pass action_dist.get_actor_actions
        bins=30, save_path=None,
    ):
    """
    cql_models_by_alpha: dict {alpha: model_entry}
    alphas_to_show: list of alpha values (must be keys in cql_models_by_alpha)
        to show, e.g. [0.0001, 0.003, 0.006, 0.01]
    clinician_actions: real clinician actions (n, 2), same processor/split
        as the models are being evaluated on
    get_actor_actions_fn: function(model_entry, processor) -> actions tensor
        (pass in action_dist.get_actor_actions to avoid re-importing here)

    Plots as small multiples -- one subplot per alpha, each with the
    clinician's real distribution repeated as a reference outline -- since
    overlaying more than 2-3 filled histograms on one axis becomes
    unreadable. Once you've identified the alphas/shapes worth featuring,
    a more selective 1-2 panel version is easy to build for publication.

    Also prints n/mean/std/median/p10/p25/p75/p90 for the clinician
    baseline and each alpha's distribution, so the trend can be described
    numerically rather than only read off the plot.
    """
    clinician_doses = get_nonzero_doses(clinician_actions, dim)

    print(f"=== {dim_name} dose distribution summary (nonzero doses only) ===")
    _print_dose_summary("Clinician (real)", clinician_doses)

    n = len(alphas_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(alphas_to_show):
        ax = axes[i]
        ax.hist(clinician_doses, bins=bins, range=(0, 1), density=True, histtype="step",
                color=CLINICIAN_COLOR, linewidth=2.5, label="Clinician (real)", zorder=5)

        if alpha not in cql_models_by_alpha:
            print(f"Warning: alpha={alpha} not found in cql_models_by_alpha, skipping.")
            continue
        model_entry = cql_models_by_alpha[alpha]
        actions = get_actor_actions_fn(model_entry, processor)
        doses = get_nonzero_doses(actions, dim)
        _print_dose_summary(f"alpha={alpha}", doses)

        color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
        ax.hist(doses, bins=bins, range=(0, 1), density=True, alpha=0.6, color=color,
                label=f"$\\alpha$={alpha}\n(n={len(doses)})")

        ax.set_xlabel(f"{dim_name} dose")
        if i == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"$\\alpha$={alpha}", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{dim_name} dose distribution, conditional on firing", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    import src.rl.action_distribution_sweep as action_dist

    clinician_actions_val = processors['val'].rl_dataset["actions"]

    fig = plot_dose_histograms(
        cql_health_score_by_alpha,
        alphas_to_show=[0.0001, 0.003, 0.006, 0.01],
        processor=processors['val'],
        clinician_actions=clinician_actions_val,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=action_dist.get_actor_actions,
        save_path="dose_histogram_fluid_by_alpha.png",
    )