"""
Violin plots of per-timestep advantage (own-critic and SARSA-critic),
one violin per seed, for the final 5-seed models -- same style as the
std-distribution violin plots, extended to advantage.

For SARSA-critic advantage, the 5 independent SARSA judges' per-timestep
advantage arrays are averaged PER TIMESTEP first (one "consensus" SARSA
advantage per timestep), then that consensus array's distribution is
shown per CQL seed -- avoids a 25-violin plot (5 CQL seeds x 5 SARSA
judges) while still using all 5 judges' information at every timestep.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

OWN_COLOR = "#c0392b"
SARSA_COLOR = "#2471a3"


def compute_advantage_distributions_by_seed(models_by_seed, processor, sarsa_models, stochastic=False):
    """
    Returns (own_advantage_by_seed, sarsa_advantage_by_seed), each a list
    of 5 per-timestep numpy arrays (one per seed). SARSA arrays are
    already averaged across the 5 SARSA judges, per timestep.

    Requires compute_model_advantage from evaluate_cql_alpha_sweep.py.
    """
    own_by_seed, sarsa_by_seed = [], []

    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"Computing advantage for seed {seed_idx + 1}/{len(models_by_seed)}...", flush=True)
        adv = compute_model_advantage(model_entry, processor, sarsa_models, stochastic=stochastic)

        own_by_seed.append(adv["own_advantage_normalized"])
        # average the 5 SARSA judges' per-timestep advantage into one consensus array
        sarsa_stack = np.stack(adv["sarsa_advantages_normalized"], axis=0)  # (5 judges, n_timesteps)
        sarsa_consensus = sarsa_stack.mean(axis=0)
        sarsa_by_seed.append(sarsa_consensus)

    return own_by_seed, sarsa_by_seed


def plot_advantage_distributions_by_seed(own_by_seed, sarsa_by_seed, log_scale=True, linthresh=None, save_path=None):
    """
    log_scale: if True, uses a symmetric log ('symlog') y-axis -- unlike a
        plain log scale, symlog handles negative values and keeps 0 at 0,
        using a linear region near zero and log-scale further out. This
        compresses the long extrema ("whisker") lines from outlier
        timesteps without hiding the sign of the advantage or distorting
        where zero sits.
    linthresh: the half-width of the linear region around 0, in the same
        units as the advantage values. If None, it's set automatically to
        the 10th percentile of |values| across all seeds combined -- i.e.
        roughly "the smallest 10% of magnitudes stay linear, everything
        larger is log-compressed". Tune this by hand if the auto choice
        looks off once you see the plot.
    """
    n_seeds = len(own_by_seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data_by_seed, title, color in [
        (axes[0], own_by_seed, "Own-critic advantage", OWN_COLOR),
        (axes[1], sarsa_by_seed, "SARSA-critic advantage\n(mean across 5 judges, per timestep)", SARSA_COLOR),
    ]:
        # NOTE: switched from violinplot to boxplot. A KDE-based violin shape
        # uses a single global bandwidth (Scott's rule), which for heavily
        # skewed/outlier-heavy data (as advantage is here -- see the long
        # extrema lines) gets inflated by the outliers and OVERSMOOTHS: the
        # sparse outlier tail can render as a visually wide "bump" while the
        # true dense cluster near the median renders thin, making the body's
        # apparent bulk misleading relative to the (correctly-placed) median
        # line. A boxplot shows quartiles/median directly from order
        # statistics -- no smoothing, so this artifact can't occur -- and
        # separates outliers into individual points rather than smearing
        # them into the body shape.
        bp = ax.boxplot(data_by_seed, showfliers=True, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2),
                         flierprops=dict(marker="o", markersize=3, alpha=0.3, markerfacecolor=color, markeredgecolor="none"))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, n_seeds + 1))
        ax.set_xticklabels([f"Seed {i+1}" for i in range(n_seeds)])
        ax.set_ylabel("Advantage (normalized, % of critic's own Q IQR)"
                       + ("\n(symlog scale)" if log_scale else ""))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        if log_scale:
            all_vals = np.concatenate(data_by_seed)
            thresh = linthresh if linthresh is not None else max(np.percentile(np.abs(all_vals), 10), 1e-6)
            ax.set_yscale("symlog", linthresh=thresh)

        print(f"\n{title}:")
        for i, vals in enumerate(data_by_seed):
            print(f"  Seed {i+1}: mean={vals.mean():.4f}, median={np.median(vals):.4f}, "
                  f"std={vals.std():.4f}, p10={np.percentile(vals,10):.4f}, p90={np.percentile(vals,90):.4f}")

    fig.suptitle("Advantage distribution, per seed (final alpha)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    own_by_seed, sarsa_by_seed = compute_advantage_distributions_by_seed(
        cql_health_score_fixed_alpha_models,
        processors['test'],
        basic_q_health_score_models,
    )
    fig = plot_advantage_distributions_by_seed(
        own_by_seed, sarsa_by_seed,
        save_path="advantage_distributions_by_seed_health_score.png",
    )