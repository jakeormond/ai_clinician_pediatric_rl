"""
Direct, model-free comparison: does giving vaso correspond to a better or
worse subsequent change in health score, among patients of comparable
severity? This works entirely from the logged reward signal and real
clinician actions -- no critic, no counterfactual model output -- so it
tests whether the pattern CQL/SARSA critics have learned is actually
visible in the raw data, or something the critic inferred beyond what a
clinician reading the paper could verify directly.

Severity binning matches the convention used elsewhere in this project
(quantile bins of health score, so each bin has comparable sample size
despite health score's skewed distribution).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

VASO_GIVEN_COLOR = "#c0392b"
NO_VASO_COLOR = "#2471a3"


def compare_reward_by_vaso(
        rewards, actions, health_score,
        n_severity_bins=20, vaso_dim=1, zero_threshold=1e-4,
    ):
    """
    rewards: (n,) -- hourly health-score change (the actual training reward)
    actions: (n, 2) -- real clinician actions
    health_score: (n,) -- that timestep's OWN (pre-dose) severity, used for binning

    Returns per-bin: mean/SEM/n for vaso-given and no-vaso groups, plus bin
    edges, so results can be inspected numerically or re-plotted.
    """
    rewards_np = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.asarray(rewards)
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    hs_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)

    vaso_given = actions_np[:, vaso_dim] > zero_threshold

    edges = np.quantile(hs_np, np.linspace(0, 1, n_severity_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_idx = np.digitize(hs_np, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_severity_bins - 1)

    results = {
        "bin_edges": edges,
        "bin_center_health_score": [],
        "mean_reward_vaso": [], "sem_reward_vaso": [], "n_vaso": [],
        "mean_reward_no_vaso": [], "sem_reward_no_vaso": [], "n_no_vaso": [],
    }

    for b in range(n_severity_bins):
        in_bin = bin_idx == b
        bin_center = hs_np[in_bin].mean() if in_bin.sum() > 0 else np.nan
        results["bin_center_health_score"].append(bin_center)

        vaso_mask = in_bin & vaso_given
        no_vaso_mask = in_bin & ~vaso_given

        for mask, prefix in [(vaso_mask, "vaso"), (no_vaso_mask, "no_vaso")]:
            n = mask.sum()
            if n > 1:
                vals = rewards_np[mask]
                results[f"mean_reward_{prefix}"].append(vals.mean())
                results[f"sem_reward_{prefix}"].append(vals.std() / np.sqrt(n))
            else:
                results[f"mean_reward_{prefix}"].append(np.nan)
                results[f"sem_reward_{prefix}"].append(np.nan)
            results[f"n_{prefix}"].append(int(n))

    for key in results:
        if key != "bin_edges":
            results[key] = np.array(results[key], dtype=float) if "n_" not in key else np.array(results[key], dtype=int)

    return results


def matched_effect_estimate(results):
    """
    Weighted-average difference (no_vaso - vaso) across bins, weighted by
    the number of vaso-given timesteps in each bin (the limiting sample
    size) -- an approximate matched-comparison effect estimate, analogous
    in spirit to the "signal" term in the earlier CQL/SARSA decomposition
    work, but computed directly from raw reward rather than a critic.
    """
    valid = ~np.isnan(results["mean_reward_vaso"]) & ~np.isnan(results["mean_reward_no_vaso"]) & (results["n_vaso"] > 1)
    diffs = results["mean_reward_no_vaso"][valid] - results["mean_reward_vaso"][valid]
    weights = results["n_vaso"][valid]
    if weights.sum() == 0:
        return np.nan, 0
    return np.average(diffs, weights=weights), weights.sum()


def plot_reward_by_vaso(results, title="", save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    x = results["bin_center_health_score"]

    ax1.errorbar(x, results["mean_reward_vaso"], yerr=results["sem_reward_vaso"],
                 marker="o", color=VASO_GIVEN_COLOR, label="Vaso given", capsize=3)
    ax1.errorbar(x, results["mean_reward_no_vaso"], yerr=results["sem_reward_no_vaso"],
                 marker="o", color=NO_VASO_COLOR, label="No vaso", capsize=3)
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Mean hourly health-score change\n(reward)")
    ax1.set_title(title or "Health-score change, vaso vs. no vaso, matched by severity", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    width = (x[1] - x[0]) * 0.4 if len(x) > 1 else 1.0
    ax2.bar(x - width/2, results["n_vaso"], width=width, color=VASO_GIVEN_COLOR, alpha=0.7, label="n (vaso)")
    ax2.bar(x + width/2, results["n_no_vaso"], width=width, color=NO_VASO_COLOR, alpha=0.7, label="n (no vaso)")
    ax2.set_yscale("log")
    ax2.set_ylabel("n per bin\n(log scale)")
    ax2.set_xlabel("Health score (severity), bin center")
    ax2.legend(frameon=False, fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    rewards = processors['test'].rl_dataset["rewards"]
    actions = processors['test'].rl_dataset["actions"]
    health_score = test_health_score  # already used elsewhere in this project

    results = compare_reward_by_vaso(rewards, actions, health_score, n_severity_bins=20)
    effect, total_n_vaso = matched_effect_estimate(results)
    print(f"Matched effect estimate (no-vaso minus vaso, weighted by n_vaso per bin): {effect:.4f}")
    print(f"Total vaso-given timesteps contributing to estimate: {total_n_vaso}")

    for b in range(len(results["bin_center_health_score"])):
        print(f"  bin center hs={results['bin_center_health_score'][b]:.1f}: "
              f"vaso mean={results['mean_reward_vaso'][b]:.3f} (n={results['n_vaso'][b]}), "
              f"no_vaso mean={results['mean_reward_no_vaso'][b]:.3f} (n={results['n_no_vaso'][b]})")

    fig = plot_reward_by_vaso(results, save_path="reward_by_vaso_matched.png")