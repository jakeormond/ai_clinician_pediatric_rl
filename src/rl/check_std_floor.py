"""
Checks whether f_log_std (and, for comparison, v_log_std) is pinned at its
clamp floor across the 5 final Stage 3 seeds (alpha=0.006), by inspecting
the raw f_std/v_std values directly (std = exp(log_std), so std at the
floor clamp corresponds to std = exp(log_std_min)).
"""

import numpy as np
import torch

LOG_STD_MIN = -2.0   # from your Actor architecture
LOG_STD_MAX = 1.0
FLOOR_STD = np.exp(LOG_STD_MIN)   # ~0.1353
CEILING_STD = np.exp(LOG_STD_MAX)  # ~2.718


def get_both_std(model_entry, processor, batch_size=1024,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Returns (f_std, v_std) as numpy arrays, shape (n_timesteps,), for every
    state in the processor's dataset.
    """
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    actor.eval()

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]

    n_total = processor.rl_dataset["states"].shape[0]
    f_std_all = torch.zeros(n_total, dtype=torch.float32)
    v_std_all = torch.zeros(n_total, dtype=torch.float32)

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
            f_std_all[start + s:start + e] = f_std.squeeze(-1).cpu()
            v_std_all[start + s:start + e] = v_std.squeeze(-1).cpu()

    return f_std_all.numpy(), v_std_all.numpy()


def check_std_floor_across_seeds(models_by_seed, processor, floor_tol=1e-3, ceiling_tol=1e-2):
    """
    models_by_seed: list of 5 model_entry tuples/objects.
    floor_tol: how close to FLOOR_STD counts as "at the floor" (accounts
        for float precision, not meant to be a strict threshold).
    ceiling_tol: how close to CEILING_STD counts as "at the ceiling" --
        looser than floor_tol since CEILING_STD (~2.718) is a larger
        number, so the same absolute float precision matters less there.

    Prints, per seed: fraction of states with f_std / v_std within
    floor_tol of FLOOR_STD or within ceiling_tol of CEILING_STD, plus
    basic summary stats. A healthy model should show these percentages
    both close to 0 -- i.e. std varies genuinely across states rather
    than being pinned at either extreme of the clamp range.
    """
    print(f"Floor std (exp(log_std_min={LOG_STD_MIN})) = {FLOOR_STD:.4f}")
    print(f"Ceiling std (exp(log_std_max={LOG_STD_MAX})) = {CEILING_STD:.4f}\n")

    all_f_std, all_v_std = [], []
    for seed_idx, model_entry in enumerate(models_by_seed):
        f_std, v_std = get_both_std(model_entry, processor)
        all_f_std.append(f_std)
        all_v_std.append(v_std)

        pct_f_at_floor = (np.abs(f_std - FLOOR_STD) < floor_tol).mean() * 100
        pct_v_at_floor = (np.abs(v_std - FLOOR_STD) < floor_tol).mean() * 100
        pct_f_at_ceiling = (np.abs(f_std - CEILING_STD) < ceiling_tol).mean() * 100
        pct_v_at_ceiling = (np.abs(v_std - CEILING_STD) < ceiling_tol).mean() * 100

        print(f"seed {seed_idx+1}: "
              f"f_std [mean={f_std.mean():.4f}, min={f_std.min():.4f}, max={f_std.max():.4f}, "
              f"%_at_floor={pct_f_at_floor:.1f}%, %_at_ceiling={pct_f_at_ceiling:.1f}%]  "
              f"v_std [mean={v_std.mean():.4f}, min={v_std.min():.4f}, max={v_std.max():.4f}, "
              f"%_at_floor={pct_v_at_floor:.1f}%, %_at_ceiling={pct_v_at_ceiling:.1f}%]")

    all_f_std = np.concatenate(all_f_std)
    all_v_std = np.concatenate(all_v_std)
    print(f"\nAcross all 5 seeds combined: "
          f"f_std mean={all_f_std.mean():.4f} "
          f"({(np.abs(all_f_std - FLOOR_STD) < floor_tol).mean()*100:.1f}% at floor, "
          f"{(np.abs(all_f_std - CEILING_STD) < ceiling_tol).mean()*100:.1f}% at ceiling), "
          f"v_std mean={all_v_std.mean():.4f} "
          f"({(np.abs(all_v_std - FLOOR_STD) < floor_tol).mean()*100:.1f}% at floor, "
          f"{(np.abs(all_v_std - CEILING_STD) < ceiling_tol).mean()*100:.1f}% at ceiling)")

    return all_f_std, all_v_std


def check_std_by_alpha(cql_models_by_alpha, processor, floor_tol=1e-3, ceiling_tol=1e-2):
    """
    Same floor/ceiling diagnostic as check_std_floor_across_seeds, but
    applied across an alpha-keyed dict (e.g. cql_health_score_by_alpha)
    rather than a list of same-alpha seeds -- lets you see whether std
    trends systematically with alpha (e.g. shrinking as conservatism
    increases), independent of the floor/ceiling question.
    """
    print(f"Floor std (exp(log_std_min={LOG_STD_MIN})) = {FLOOR_STD:.4f}")
    print(f"Ceiling std (exp(log_std_max={LOG_STD_MAX})) = {CEILING_STD:.4f}\n")

    alphas = sorted(cql_models_by_alpha.keys())
    results = {}

    for alpha in alphas:
        model_entry = cql_models_by_alpha[alpha]
        f_std, v_std = get_both_std(model_entry, processor)

        pct_f_at_floor = (np.abs(f_std - FLOOR_STD) < floor_tol).mean() * 100
        pct_v_at_floor = (np.abs(v_std - FLOOR_STD) < floor_tol).mean() * 100
        pct_f_at_ceiling = (np.abs(f_std - CEILING_STD) < ceiling_tol).mean() * 100
        pct_v_at_ceiling = (np.abs(v_std - CEILING_STD) < ceiling_tol).mean() * 100

        results[alpha] = {
            "f_std_mean": f_std.mean(), "f_std_min": f_std.min(), "f_std_max": f_std.max(),
            "f_pct_floor": pct_f_at_floor, "f_pct_ceiling": pct_f_at_ceiling,
            "v_std_mean": v_std.mean(), "v_std_min": v_std.min(), "v_std_max": v_std.max(),
            "v_pct_floor": pct_v_at_floor, "v_pct_ceiling": pct_v_at_ceiling,
        }

        print(f"alpha={alpha}: "
              f"f_std [mean={f_std.mean():.4f}, min={f_std.min():.4f}, max={f_std.max():.4f}, "
              f"%floor={pct_f_at_floor:.1f}%, %ceiling={pct_f_at_ceiling:.1f}%]  "
              f"v_std [mean={v_std.mean():.4f}, min={v_std.min():.4f}, max={v_std.max():.4f}, "
              f"%floor={pct_v_at_floor:.1f}%, %ceiling={pct_v_at_ceiling:.1f}%]")

    return results


def plot_std_by_alpha(results, log_x=True, save_path=None):
    """
    Plots f_std_mean / v_std_mean vs. alpha, with floor/ceiling reference
    lines. log_x=True (default) uses a log-scale x-axis, since alpha
    typically spans several orders of magnitude -- set False if alpha=0
    is present in `results` (log(0) is undefined; alpha=0 is silently
    dropped from the plot when log_x=True, with a printed note).
    """
    import matplotlib.pyplot as plt

    alphas = sorted(results.keys())
    if log_x and 0 in alphas:
        print("Note: alpha=0 excluded from plot (log scale can't show it).")
        alphas = [a for a in alphas if a != 0]

    f_means = [results[a]["f_std_mean"] for a in alphas]
    v_means = [results[a]["v_std_mean"] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, f_means, marker="o", color="#2471a3", label="Fluid std (mean)")
    ax.plot(alphas, v_means, marker="o", color="#c0392b", label="Vaso std (mean)")
    ax.axhline(FLOOR_STD, color="grey", linestyle="--", linewidth=1, label="Floor")
    ax.axhline(CEILING_STD, color="grey", linestyle=":", linewidth=1, label="Ceiling")
    ax.set_xlabel(r"cql_$\alpha$" + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Mean std across states")
    ax.set_title("Actor magnitude std vs. alpha", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)  # prevent a stray cached figure from being redisplayed by a later plt call in the same session
    return fig


if __name__ == "__main__":
    all_f_std, all_v_std = check_std_floor_across_seeds(
        cql_health_score_fixed_alpha_models,
        processors['val'],
    )

    results_std_hs = check_std_by_alpha(cql_health_score_by_alpha, processors['val'])
    plot_std_by_alpha(results_std_hs, save_path="std_by_alpha_health_score.png")

    results_std_term = check_std_by_alpha(cql_terminal_by_alpha, processors['val'])
    plot_std_by_alpha(results_std_term, save_path="std_by_alpha_terminal.png")


def plot_std_distributions_by_seed(models_by_seed, processor, save_path=None):
    """
    Violin plot of f_std / v_std distributions, one violin per seed, for
    the final fixed-alpha 5-seed models -- a fuller check than the
    floor/ceiling percentages alone: shows the actual shape of each seed's
    std distribution (e.g. genuinely bimodal vs. a single well-behaved
    mode), and whether that shape is consistent across seeds.
    """
    import matplotlib.pyplot as plt

    n_seeds = len(models_by_seed)
    f_std_by_seed, v_std_by_seed = [], []

    for model_entry in models_by_seed:
        f_std, v_std = get_both_std(model_entry, processor)
        f_std_by_seed.append(f_std)
        v_std_by_seed.append(v_std)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data_by_seed, title, color in [
        (axes[0], f_std_by_seed, "Fluid std", "#2471a3"),
        (axes[1], v_std_by_seed, "Vaso std", "#c0392b"),
    ]:
        parts = ax.violinplot(data_by_seed, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        ax.axhline(FLOOR_STD, color="grey", linestyle="--", linewidth=1, label="Floor")
        ax.axhline(CEILING_STD, color="grey", linestyle=":", linewidth=1, label="Ceiling")
        ax.set_xticks(range(1, n_seeds + 1))
        ax.set_xticklabels([f"Seed {i+1}" for i in range(n_seeds)])
        ax.set_ylabel("std across states")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Actor magnitude std distribution, per seed (final alpha)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)
    return fig