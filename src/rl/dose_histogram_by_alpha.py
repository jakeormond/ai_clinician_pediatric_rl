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
        cql_models_by_alpha, processor, clinician_actions,
        alphas_to_show=None,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=None,  # pass action_dist.get_actor_actions
        bins=30, n_cols=5, save_path=None,
    ):
    """
    cql_models_by_alpha: dict {alpha: model_entry}
    alphas_to_show: list of alpha values (must be keys in cql_models_by_alpha)
        to show, e.g. [0.0001, 0.003, 0.006, 0.01]. If None (default),
        plots ALL alphas in cql_models_by_alpha, sorted ascending.
    clinician_actions: real clinician actions (n, 2), same processor/split
        as the models are being evaluated on
    get_actor_actions_fn: function(model_entry, processor) -> actions tensor
        (pass in action_dist.get_actor_actions to avoid re-importing here)
    n_cols: number of panels per row in the grid (default 5) -- rows are
        added automatically to fit however many alphas are being shown.

    Plots as small multiples in a grid -- one subplot per alpha, each with
    the clinician's real distribution repeated as a reference outline --
    since overlaying more than 2-3 filled histograms on one axis becomes
    unreadable.

    Also prints n/mean/std/median/p10/p25/p75/p90 for the clinician
    baseline and each alpha's distribution, so the trend can be described
    numerically rather than only read off the plot.
    """
    if alphas_to_show is None:
        alphas_to_show = sorted(cql_models_by_alpha.keys())

    clinician_doses = get_nonzero_doses(clinician_actions, dim)

    print(f"=== {dim_name} dose distribution summary (nonzero doses only) ===")
    _print_dose_summary("Clinician (real)", clinician_doses)

    n = len(alphas_to_show)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).flatten()  # flatten regardless of 1D/2D grid shape for uniform indexing

    for i, alpha in enumerate(alphas_to_show):
        ax = axes[i]
        ax.hist(clinician_doses, bins=bins, range=(0, 1), density=True, histtype="step",
                color=CLINICIAN_COLOR, linewidth=2, label="Clinician (real)", zorder=5)

        if alpha not in cql_models_by_alpha:
            print(f"Warning: alpha={alpha} not found in cql_models_by_alpha, skipping.")
            continue
        model_entry = cql_models_by_alpha[alpha]
        actions = get_actor_actions_fn(model_entry, processor)
        doses = get_nonzero_doses(actions, dim)
        _print_dose_summary(f"alpha={alpha}", doses)

        color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
        ax.hist(doses, bins=bins, range=(0, 1), density=True, alpha=0.6, color=color,
                label=f"n={len(doses)}")

        ax.set_title(f"$\\alpha$={alpha}", fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)

    # only label axes on the outer edge of the grid, to reduce clutter
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= n:
                axes[idx].axis("off")  # hide unused panels if grid isn't fully filled
                continue
            if col == 0:
                axes[idx].set_ylabel("Density")
            if row == n_rows - 1 or idx + n_cols >= n:  # bottom row, or last row of a ragged grid
                axes[idx].set_xlabel(f"{dim_name} dose")

    fig.suptitle(f"{dim_name} dose distribution, conditional on firing", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_dose_histograms_at_clinician_zero(
        cql_models_by_alpha, processor, clinician_actions,
        alphas_to_show=None,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=None,
        zero_threshold=1e-4, bins=30, n_cols=5, save_path=None,
    ):
    """
    Companion to plot_dose_histograms, restricted to timesteps where the
    CLINICIAN gave zero dose. Shows the full model dose distribution at
    these states (including zero), rather than a nonzero-conditional
    shape -- the question here is different: not "what shape does the
    dose take when given", but "does the model agree treatment should be
    withheld at all, and if it disagrees, what does it recommend instead".

    Prints, per alpha: fraction of clinician-zero states where the model
    ALSO recommends ~zero (agreement rate), and the distribution of doses
    among the states where it disagrees (recommends nonzero).
    """
    if alphas_to_show is None:
        alphas_to_show = sorted(cql_models_by_alpha.keys())

    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    clinician_zero_mask = clinician_actions_np[:, dim] <= zero_threshold
    n_clinician_zero = clinician_zero_mask.sum()

    print(f"=== {dim_name}: model dose distribution AT clinician-zero states "
          f"(n={n_clinician_zero}, {n_clinician_zero / len(clinician_actions_np) * 100:.1f}% of all states) ===")

    n = len(alphas_to_show)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for i, alpha in enumerate(alphas_to_show):
        ax = axes[i]

        if alpha not in cql_models_by_alpha:
            print(f"Warning: alpha={alpha} not found in cql_models_by_alpha, skipping.")
            continue
        model_entry = cql_models_by_alpha[alpha]
        actions = get_actor_actions_fn(model_entry, processor)
        actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)

        model_dose_at_clinician_zero = actions_np[clinician_zero_mask, dim]
        agree_rate = (model_dose_at_clinician_zero <= zero_threshold).mean()
        disagree_doses = model_dose_at_clinician_zero[model_dose_at_clinician_zero > zero_threshold]

        if len(disagree_doses) > 0:
            print(f"  alpha={alpha}: agrees (recommends ~0) at {agree_rate*100:.1f}% of clinician-zero states; "
                  f"among the {len(disagree_doses)} disagreements, mean dose={disagree_doses.mean():.4f}, "
                  f"median={np.median(disagree_doses):.4f}")
        else:
            print(f"  alpha={alpha}: agrees (recommends ~0) at {agree_rate*100:.1f}% of clinician-zero states "
                  f"(no disagreements)")

        # plot the FULL distribution (including zero) of what the model recommends
        # at these states -- a spike at 0 in this plot IS agreement, not a problem
        ax.hist(model_dose_at_clinician_zero, bins=bins, range=(0, 1), density=True,
                alpha=0.6, color=ALPHA_COLORS[i % len(ALPHA_COLORS)],
                label=f"agree={agree_rate*100:.0f}%")

        ax.set_title(f"$\\alpha$={alpha}", fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= n:
                axes[idx].axis("off")
                continue
            if col == 0:
                axes[idx].set_ylabel("Density")
            if row == n_rows - 1 or idx + n_cols >= n:
                axes[idx].set_xlabel(f"{dim_name} dose")

    fig.suptitle(f"{dim_name} dose recommended by model, AT states where clinician gave zero",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_combined_dose_distribution_overlay(
        models_by_key, processor, clinician_actions,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=None,
        bins=30, title="", save_path=None,
    ):
    """
    Complementary view to plot_dose_histograms: instead of one small-
    multiple panel per key (seed or alpha), overlays the clinician's real
    distribution (bold, black step outline) with EVERY model's
    distribution (thin, colored step lines, one per key) on a SINGLE axis.
    Easier to see at a glance both how closely each model matches the
    real shape, and how consistent multiple seeds/runs are with each
    other -- at the cost of being harder to read if models diverge a lot
    from one another (in which case plot_dose_histograms' small multiples
    remain the clearer choice).
 
    models_by_key: dict {key: model_entry}, e.g. {0: bc_seed_0_entry, ...}
    get_actor_actions_fn: function(model_entry, processor, batch_size=,
        device=) -> actions tensor. Required -- pass e.g.
        eval_bc.get_bc_actor_actions for BC actors, or
        action_distribution_sweep.get_actor_actions for CQL actors.
    """
    if get_actor_actions_fn is None:
        raise ValueError("get_actor_actions_fn is required (no default -- actor call "
                          "signatures differ between model types, e.g. BC vs CQL).")
 
    clinician_doses = get_nonzero_doses(clinician_actions, dim)
 
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(clinician_doses, bins=bins, range=(0, 1), density=True, histtype="step",
            color=CLINICIAN_COLOR, linewidth=3, label=f"Clinician (real, n={len(clinician_doses)})", zorder=10)
 
    keys = sorted(models_by_key.keys())
    for i, key in enumerate(keys):
        model_entry = models_by_key[key]
        actions = get_actor_actions_fn(model_entry, processor)
        doses = get_nonzero_doses(actions, dim)
        color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
        ax.hist(doses, bins=bins, range=(0, 1), density=True, histtype="step",
                color=color, linewidth=1.5, alpha=0.85, label=f"Seed {key} (n={len(doses)})")
 
    ax.set_xlabel(f"{dim_name} dose")
    ax.set_ylabel("Density")
    ax.set_title(title or f"{dim_name} dose distribution: all seeds vs. clinician (nonzero doses only)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
 
    plt.close(fig)
    return fig


if __name__ == "__main__":
    import src.rl.action_distribution_sweep as action_dist

    clinician_actions_val = processors['val'].rl_dataset["actions"]

    # plot ALL alphas currently in cql_health_score_by_alpha, 5 per row
    fig = plot_dose_histograms(
        cql_health_score_by_alpha,
        processor=processors['val'],
        clinician_actions=clinician_actions_val,
        dim=0, dim_name="Fluid",
        get_actor_actions_fn=action_dist.get_actor_actions,
        n_cols=5,
        save_path="dose_histogram_fluid_by_alpha.png",
    )