"""
Direct inspection of the actor's raw fluid magnitude parameters
(f_raw_mean, f_std -- the pre-sigmoid Gaussian the fluid dose is drawn
from) across alpha values, to check whether bimodal/extreme dosing at low
alpha is explained by f_raw_mean drifting to extreme (saturating) values,
independent of any CQL-loss or Q-landscape explanation.

Requires calling the actor directly (not through generate_model_outputs,
which only returns final actions) to access the internal distribution
parameters via return_dist=True.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

ALPHA_COLORS = ["#7d3c98", "#2471a3", "#229954", "#c0392b"]


def get_raw_magnitude_params(model_entry, processor, batch_size=1024,
                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Runs the actor in deterministic mode with return_dist=True to pull out
    f_raw_mean and f_std for every state in the processor's dataset.
    Returns (f_raw_mean, f_std) as numpy arrays, shape (n_timesteps,).
    """
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    actor.eval()

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]

    n_total = processor.rl_dataset["states"].shape[0]
    f_raw_mean_all = torch.zeros(n_total, dtype=torch.float32)
    f_std_all = torch.zeros(n_total, dtype=torch.float32)

    for stay_id in stay_ids:
        stay_idx = np.where(stay_ids == stay_id)[0][0]
        start, end = traj_start_inds[stay_idx], traj_start_inds[stay_idx + 1]
        states = torch.as_tensor(processor.rl_dataset["states"][start:end], dtype=torch.float32).to(device)

        n_batches = int(np.ceil(states.shape[0] / batch_size))
        for i in range(n_batches):
            s, e = i * batch_size, min((i + 1) * batch_size, states.shape[0])
            states_batch = states[s:e]
            with torch.no_grad():
                _, _, _, (f_p, v_p, f_raw_mean, v_raw_mean, f_std, v_std) = actor(
                    states_batch, deterministic=True, return_dist=True
                )
            f_raw_mean_all[start + s:start + e] = f_raw_mean.squeeze(-1).cpu()
            f_std_all[start + s:start + e] = f_std.squeeze(-1).cpu()

    return f_raw_mean_all.numpy(), f_std_all.numpy()


def summarize_raw_params(f_raw_mean, f_std, label=""):
    print(f"  {label}: f_raw_mean [mean={f_raw_mean.mean():.3f}, std={f_raw_mean.std():.3f}, "
          f"min={f_raw_mean.min():.3f}, max={f_raw_mean.max():.3f}, "
          f"p10={np.percentile(f_raw_mean, 10):.3f}, p90={np.percentile(f_raw_mean, 90):.3f}]  "
          f"f_std [mean={f_std.mean():.3f}, min={f_std.min():.3f}, max={f_std.max():.3f}]")


def plot_raw_params_by_alpha(cql_models_by_alpha, alphas_to_show, processor, save_path=None):
    """
    Small multiples: one histogram of f_raw_mean per alpha, so you can see
    directly whether the raw pre-sigmoid mean is bimodal/saturating toward
    extreme values at low alpha, and how that changes across the sweep.
    """
    n = len(alphas_to_show)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    print("=== f_raw_mean / f_std summary across alpha ===")
    for i, alpha in enumerate(alphas_to_show):
        ax = axes[i]
        if alpha not in cql_models_by_alpha:
            print(f"Warning: alpha={alpha} not found, skipping.")
            continue
        model_entry = cql_models_by_alpha[alpha]
        f_raw_mean, f_std = get_raw_magnitude_params(model_entry, processor)
        summarize_raw_params(f_raw_mean, f_std, label=f"alpha={alpha}")

        color = ALPHA_COLORS[i % len(ALPHA_COLORS)]
        ax.hist(f_raw_mean, bins=40, color=color, alpha=0.7)
        ax.axvline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_xlabel("f_raw_mean (pre-sigmoid)")
        if i == 0:
            ax.set_ylabel("Count")
        ax.set_title(f"$\\alpha$={alpha}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Raw fluid magnitude mean (pre-sigmoid) across alpha", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    fig = plot_raw_params_by_alpha(
        cql_health_score_by_alpha,
        alphas_to_show=[0.0001, 0.003, 0.006, 0.01],
        processor=processors['val'],
        save_path="fluid_raw_params_by_alpha.png",
    )