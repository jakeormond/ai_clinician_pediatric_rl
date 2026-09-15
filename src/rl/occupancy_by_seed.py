"""
Occupancy support for the final 5-seed models: fraction of each seed's
chosen actions that are well-supported (joint severity x fluid x vaso
count >= min_count in training data), plus a per-timestep view of the
underlying occupancy counts themselves.

Reuses compute_occupancy_support_of_actions and get_actor_actions from
evaluate_cql_alpha_sweep.py / action_distribution_sweep.py.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

OCC_COLOR = "#229954"


def compute_occupancy_by_seed(
        models_by_seed, processor, test_health_score, training_actions, training_health_score,
        n_severity_bins=20, n_action_bins=10, min_count=10,
    ):
    """
    Returns:
        support_frac_by_seed: list of 5 scalars (fraction well-supported per seed)
        counts_by_seed: list of 5 per-timestep occupancy-count arrays (for the
            box-plot / distributional view -- note these are RAW counts, not
            fractions, so the plot shows "how many training examples backed
            this exact recommendation" rather than a 0/1 well-supported flag)
    """
    support_frac_by_seed, counts_by_seed = [], []

    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"Computing occupancy for seed {seed_idx + 1}/{len(models_by_seed)}...", flush=True)
        actions = get_actor_actions(model_entry, processor)
        support_frac, counts = compute_occupancy_support_of_actions(
            actions, test_health_score, training_actions, training_health_score,
            n_severity_bins=n_severity_bins, n_action_bins=n_action_bins, min_count=min_count,
        )
        support_frac_by_seed.append(support_frac)
        counts_by_seed.append(counts)
        print(f"  seed {seed_idx + 1}: {support_frac*100:.1f}% well-supported "
              f"(min_count={min_count}), median count={np.median(counts):.0f}")

    return support_frac_by_seed, counts_by_seed


def plot_occupancy_by_seed(support_frac_by_seed, counts_by_seed, min_count=10, log_scale=True, save_path=None):
    n_seeds = len(support_frac_by_seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # panel A: single well-supported-fraction summary per seed, as a bar
    axes[0].bar(range(1, n_seeds + 1), [f * 100 for f in support_frac_by_seed], color=OCC_COLOR, alpha=0.8)
    axes[0].set_xticks(range(1, n_seeds + 1))
    axes[0].set_xticklabels([f"Seed {i+1}" for i in range(n_seeds)])
    axes[0].set_ylabel(f"% of actions well-supported (n\u2265{min_count})")
    axes[0].set_title("Occupancy support, per seed", fontsize=11, fontweight="bold", pad=20)
    axes[0].set_ylim(0, 108)
    axes[0].spines[["top", "right"]].set_visible(False)
    for i, f in enumerate(support_frac_by_seed):
        axes[0].text(i + 1, f * 100 + 1.5, f"{f*100:.1f}%", ha="center", fontsize=9)

    # panel B: distribution of the underlying raw occupancy counts, per seed
    # (boxplot, not violin -- same reasoning as the advantage plots: counts
    # are heavily right-skewed/outlier-prone, KDE would be misleading here too)
    bp = axes[1].boxplot(counts_by_seed, showfliers=True, patch_artist=True,
                          medianprops=dict(color="black", linewidth=2),
                          flierprops=dict(marker="o", markersize=3, alpha=0.2, markerfacecolor=OCC_COLOR, markeredgecolor="none"))
    for patch in bp["boxes"]:
        patch.set_facecolor(OCC_COLOR)
        patch.set_alpha(0.5)
    axes[1].axhline(min_count, color="grey", linestyle="--", linewidth=1, label=f"min_count={min_count}")
    axes[1].set_xticks(range(1, n_seeds + 1))
    axes[1].set_xticklabels([f"Seed {i+1}" for i in range(n_seeds)])
    axes[1].set_ylabel("Training-data occupancy count\nat chosen (severity, fluid, vaso) bin")
    axes[1].set_title("Distribution of occupancy counts, per seed", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    if log_scale:
        axes[1].set_yscale("symlog", linthresh=1)  # counts are non-negative; symlog just gives log-like
        # compression for large counts while keeping 0 representable (plain log can't show 0)

    fig.suptitle("Occupancy support, per seed (final alpha)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    support_frac_by_seed, counts_by_seed = compute_occupancy_by_seed(
        cql_health_score_fixed_alpha_models,
        processors['test'],
        test_health_score,
        training_actions,
        training_health_score,
    )
    fig = plot_occupancy_by_seed(
        support_frac_by_seed, counts_by_seed,
        save_path="occupancy_by_seed_health_score.png",
    )