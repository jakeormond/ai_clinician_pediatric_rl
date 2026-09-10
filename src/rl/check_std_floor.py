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


def check_std_floor_across_seeds(models_by_seed, processor, floor_tol=1e-3):
    """
    models_by_seed: list of 5 model_entry tuples/objects.
    floor_tol: how close to FLOOR_STD counts as "at the floor" (accounts
        for float precision, not meant to be a strict threshold).

    Prints, per seed: fraction of states with f_std / v_std within
    floor_tol of FLOOR_STD, plus basic summary stats.
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

        print(f"seed {seed_idx+1}: "
              f"f_std [mean={f_std.mean():.4f}, min={f_std.min():.4f}, max={f_std.max():.4f}, "
              f"%_at_floor={pct_f_at_floor:.1f}%]  "
              f"v_std [mean={v_std.mean():.4f}, min={v_std.min():.4f}, max={v_std.max():.4f}, "
              f"%_at_floor={pct_v_at_floor:.1f}%]")

    all_f_std = np.concatenate(all_f_std)
    all_v_std = np.concatenate(all_v_std)
    print(f"\nAcross all 5 seeds combined: "
          f"f_std mean={all_f_std.mean():.4f} ({(np.abs(all_f_std - FLOOR_STD) < floor_tol).mean()*100:.1f}% at floor), "
          f"v_std mean={all_v_std.mean():.4f} ({(np.abs(all_v_std - FLOOR_STD) < floor_tol).mean()*100:.1f}% at floor)")

    return all_f_std, all_v_std


if __name__ == "__main__":
    all_f_std, all_v_std = check_std_floor_across_seeds(
        cql_health_score_fixed_alpha_models,
        processors['val'],
    )