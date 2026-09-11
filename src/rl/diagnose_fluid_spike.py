"""
Diagnoses the discrete population of states producing the fluid dose spike
near 1.0, by comparing spike vs. non-spike states on: severity (health
score), gate confidence (f_p), and real clinician fluid dose at those same
states. This is meant to distinguish three hypotheses:

  1. Spike concentrated at a specific severity range -> a real feature of
     the critic's Q-landscape at that severity (possible confound or
     genuine effect), not an actor-loss problem.
  2. Spike states have f_p extremely close to 1.0 (numerical precision in
     the smooth weight going effectively-but-not-exactly to zero) -> a
     numerical fix, not a conceptual one.
  3. Spike doesn't line up cleanly with either -> likely a critic-side /
     CQL-sampling artifact (e.g. vaso_only_actions or zero_occupancy_actions
     biasing Q upward for high-fluid-dose regions), not fixable via the
     actor's loss at all.
"""

import numpy as np
import torch


def get_raw_params_and_gate(model_entry, processor, batch_size=1024,
                              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Returns f_raw_mean, f_p (both shape (n_timesteps,)) for every state in
    the processor's dataset, in the SAME order as processor.rl_dataset --
    so they align directly with test_health_score / clinician_actions
    indexed the same way.
    """
    actor = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry
    actor.eval()

    traj_end_inds = np.where(processor.rl_dataset["dones"] == 1)[0]
    traj_start_inds = np.concatenate(([0], traj_end_inds + 1))
    stay_ids = processor.dynamic_deephit_dataset["stay_ids"]

    n_total = processor.rl_dataset["states"].shape[0]
    f_raw_mean_all = torch.zeros(n_total, dtype=torch.float32)
    f_p_all = torch.zeros(n_total, dtype=torch.float32)

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
            f_p_all[start + s:start + e] = f_p.squeeze(-1).cpu()

    return f_raw_mean_all.numpy(), f_p_all.numpy()


def diagnose_fluid_spike(f_raw_mean, f_p, health_score, clinician_actions, spike_threshold=5.0):
    """
    f_raw_mean, f_p: from get_raw_params_and_gate, shape (n,)
    health_score: shape (n,) -- same ordering as processor.rl_dataset
    clinician_actions: shape (n, 2) -- same ordering, real clinician doses
    spike_threshold: f_raw_mean above this is classified as "in the spike"
        -- set based on where the histogram shows a clear gap between the
        main mass and the spike (5.0 is a starting guess; adjust after
        looking at the actual f_raw_mean histogram).
    """
    health_score_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)
    actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)

    spike_mask = f_raw_mean > spike_threshold
    n_spike = spike_mask.sum()
    n_total = len(f_raw_mean)

    print(f"Spike threshold: f_raw_mean > {spike_threshold}")
    print(f"Fraction of states in spike: {n_spike / n_total:.4f}  ({n_spike} of {n_total})")
    print()

    print("--- Health score (severity) ---")
    print(f"  Spike states:     mean={health_score_np[spike_mask].mean():.2f}, "
          f"std={health_score_np[spike_mask].std():.2f}, "
          f"p10={np.percentile(health_score_np[spike_mask], 10):.2f}, "
          f"p90={np.percentile(health_score_np[spike_mask], 90):.2f}")
    print(f"  Non-spike states: mean={health_score_np[~spike_mask].mean():.2f}, "
          f"std={health_score_np[~spike_mask].std():.2f}, "
          f"p10={np.percentile(health_score_np[~spike_mask], 10):.2f}, "
          f"p90={np.percentile(health_score_np[~spike_mask], 90):.2f}")
    print()

    print("--- Gate confidence (f_p) ---")
    print(f"  Spike states:     mean={f_p[spike_mask].mean():.6f}, min={f_p[spike_mask].min():.6f}, "
          f"max={f_p[spike_mask].max():.6f}")
    print(f"  Non-spike states: mean={f_p[~spike_mask].mean():.6f}, min={f_p[~spike_mask].min():.6f}, "
          f"max={f_p[~spike_mask].max():.6f}")
    print(f"  Spike states within 1e-4 of f_p=1.0: {(f_p[spike_mask] > 1 - 1e-4).mean():.4f}")
    print(f"  Spike states within 1e-6 of f_p=1.0: {(f_p[spike_mask] > 1 - 1e-6).mean():.4f}")
    print()

    print("--- Real clinician fluid dose, at these same states ---")
    print(f"  Spike states:     mean={actions_np[spike_mask, 0].mean():.4f}, "
          f"median={np.median(actions_np[spike_mask, 0]):.4f}")
    print(f"  Non-spike states: mean={actions_np[~spike_mask, 0].mean():.4f}, "
          f"median={np.median(actions_np[~spike_mask, 0]):.4f}")
    print()

    print("--- Real clinician VASO dose, at these same states (checking for cross-drug pattern) ---")
    print(f"  Spike states:     mean={actions_np[spike_mask, 1].mean():.4f}, "
          f"fire_rate={(actions_np[spike_mask, 1] > 1e-4).mean():.4f}")
    print(f"  Non-spike states: mean={actions_np[~spike_mask, 1].mean():.4f}, "
          f"fire_rate={(actions_np[~spike_mask, 1] > 1e-4).mean():.4f}")

    return spike_mask


if __name__ == "__main__":
    f_raw_mean, f_p = get_raw_params_and_gate(
        cql_health_score_fixed_alpha_models[0],  # adjust index/variable as needed
        processors['test'],
    )
    clinician_actions_test = processors['test'].rl_dataset["actions"]

    spike_mask = diagnose_fluid_spike(
        f_raw_mean, f_p, test_health_score, clinician_actions_test,
        spike_threshold=5.0,
    )