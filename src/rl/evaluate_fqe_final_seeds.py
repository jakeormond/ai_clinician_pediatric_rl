"""
Advantage analysis for the 5 final FQE models: Q_FQE(target-policy action)
vs. Q_FQE(clinician action), for each of the 5 (CQL seed, FQE seed) pairs.

IMPORTANT: unlike SARSA (a policy-agnostic critic that can judge ANY
policy's actions after the fact), each FQE critic was fit specifically to
estimate the value of ONE target policy -- the CQL actor plugged into its
own Bellman backup during FQE training (see cql_with_logprobs.load_trained_actor
and basic_q.Q_Learner's target_policy_actor). FQE seed i's critic is only
theoretically valid for evaluating CQL seed i's policy specifically -- it
must NOT be used to score any of the other 4 CQL seeds' actions. This file
enforces that pairing by matching on each model's own recorded
config_manager.training_config['seed'], rather than trusting list order.
"""

import os
import numpy as np
import torch


def get_fqe_critic(model_entry):
    """FQE tuples are (critic, config_manager, hyperparams) -- same shape as SARSA."""
    return model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry


def get_seed_from_checkpoint(model_dir, device="cpu"):
    """
    Reads the training seed directly from a model's OWN saved checkpoint
    (final_model.pt's training_config), rather than from a possibly-SHARED
    config_manager.pkl -- confirmed necessary because config_manager.pkl is
    sometimes saved once per SWEEP (shared across all seed subfolders)
    rather than once per individual run, in which case every model in a
    seed sweep would incorrectly report the SAME (template/launch-time)
    seed value if read from config_manager. The checkpoint itself is
    always run-specific, so this is the reliable source of truth.
    """
    final_model_path = os.path.join(model_dir, "final_model.pt")
    checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    if "training_config" not in checkpoint or "seed" not in checkpoint["training_config"]:
        raise KeyError(f"No training_config['seed'] found in checkpoint at {final_model_path}. "
                        f"Available checkpoint keys: {list(checkpoint.keys())}")
    return checkpoint["training_config"]["seed"]


def match_cql_fqe_pairs(cql_models, fqe_models, cql_model_dirs, fqe_model_dirs):
    """
    Matches each CQL model to the FQE model trained to evaluate IT
    specifically, verified via the seed recorded in each model's OWN
    checkpoint (NOT config_manager.pkl, which may be shared across a
    whole seed sweep -- see get_seed_from_checkpoint). Raises if any CQL
    model has no matching FQE model, or vice versa.

    cql_models/fqe_models and cql_model_dirs/fqe_model_dirs must be in
    the SAME order (i.e. cql_models[i] was loaded from cql_model_dirs[i]) --
    this matches how you already load them in your notebook.

    Returns a list of (cql_model_entry, fqe_model_entry, seed) tuples,
    sorted by seed.
    """
    cql_seeds = [get_seed_from_checkpoint(d) for d in cql_model_dirs]
    fqe_seeds = [get_seed_from_checkpoint(d) for d in fqe_model_dirs]

    print(f"CQL model dirs -> seeds: {list(zip([os.path.basename(d) for d in cql_model_dirs], cql_seeds))}")
    print(f"FQE model dirs -> seeds: {list(zip([os.path.basename(d) for d in fqe_model_dirs], fqe_seeds))}")

    cql_by_seed = dict(zip(cql_seeds, cql_models))
    fqe_by_seed = dict(zip(fqe_seeds, fqe_models))

    cql_seed_set = set(cql_seeds)
    fqe_seed_set = set(fqe_seeds)

    if len(cql_seed_set) != len(cql_models):
        raise ValueError(f"CQL model_dirs do not have distinct seeds: {cql_seeds}")
    if len(fqe_seed_set) != len(fqe_models):
        raise ValueError(f"FQE model_dirs do not have distinct seeds: {fqe_seeds}")

    if cql_seed_set != fqe_seed_set:
        raise ValueError(
            f"CQL and FQE seed sets don't match. "
            f"CQL seeds: {sorted(cql_seed_set)}, FQE seeds: {sorted(fqe_seed_set)}. "
            f"CQL-only: {sorted(cql_seed_set - fqe_seed_set)}, FQE-only: {sorted(fqe_seed_set - cql_seed_set)}"
        )

    pairs = [(cql_by_seed[s], fqe_by_seed[s], s) for s in sorted(cql_seed_set)]
    print(f"Matched {len(pairs)} (CQL, FQE) pairs by checkpoint seed: {sorted(cql_seed_set)}")
    return pairs


def get_cql_actor(cql_model_entry):
    """CQL tuples are (actor, critic, config_manager, hyperparams)."""
    return cql_model_entry[0] if isinstance(cql_model_entry, (list, tuple)) else cql_model_entry


def compute_fqe_advantage(
        cql_model_entry, fqe_model_entry, processor, clinician_actions,
        batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu",
    ):
    """
    For ONE matched (CQL seed, FQE seed) pair: computes Q_FQE at the real
    clinician action and at the CQL actor's own (deterministic) chosen
    action, for every state in processor's dataset.

    Returns a dict with raw and IQR-normalized advantage (normalized by
    this FQE critic's own Q-IQR under real clinician actions, matching
    the normalization convention used throughout this project's SARSA/CQL
    advantage analyses), plus the two raw Q arrays for further inspection.
    """
    fqe_critic = get_fqe_critic(fqe_model_entry)
    cql_actor = get_cql_actor(cql_model_entry)
    fqe_critic.eval()
    cql_actor.eval()

    states = torch.as_tensor(processor.rl_dataset["states"], dtype=torch.float32)
    clinician_actions_t = clinician_actions if torch.is_tensor(clinician_actions) else torch.as_tensor(clinician_actions, dtype=torch.float32)

    n = states.shape[0]
    q_clinician = torch.zeros(n)
    q_cql = torch.zeros(n)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            s = states[i:i+batch_size].to(device)
            a_clinician = clinician_actions_t[i:i+batch_size].to(device)

            cql_action, _, _ = cql_actor(s)  # deterministic action, matching how CQL's actor is evaluated elsewhere

            q1_c, q2_c = fqe_critic(s, a_clinician)
            q1_p, q2_p = fqe_critic(s, cql_action)

            q_clinician[i:i+batch_size] = ((q1_c + q2_c) / 2.0).squeeze(-1).cpu()
            q_cql[i:i+batch_size] = ((q1_p + q2_p) / 2.0).squeeze(-1).cpu()

    advantage = (q_cql - q_clinician).numpy()
    q_clinician_np = q_clinician.numpy()

    q_iqr = np.percentile(q_clinician_np, 75) - np.percentile(q_clinician_np, 25)
    advantage_normalized = advantage / q_iqr if q_iqr > 0 else advantage

    return {
        "advantage": advantage,
        "advantage_normalized": advantage_normalized,
        "q_clinician": q_clinician_np,
        "q_cql": q_cql.numpy(),
        "q_iqr": q_iqr,
    }


def run_matched_seed_fqe_advantage(cql_models, fqe_models, cql_model_dirs, fqe_model_dirs,
                                     processor, clinician_actions):
    """
    Runs compute_fqe_advantage for every correctly-matched (CQL, FQE) seed
    pair, printing a summary per seed and returning the full per-seed
    results for further analysis/plotting.
    """
    pairs = match_cql_fqe_pairs(cql_models, fqe_models, cql_model_dirs, fqe_model_dirs)
    results = []

    for cql_model_entry, fqe_model_entry, seed in pairs:
        print(f"\n=== Seed {seed} ===")
        result = compute_fqe_advantage(cql_model_entry, fqe_model_entry, processor, clinician_actions)
        result["seed"] = seed
        results.append(result)

        adv = result["advantage"]
        adv_norm = result["advantage_normalized"]
        print(f"  Q IQR (clinician actions): {result['q_iqr']:.4f}")
        print(f"  Raw advantage:        mean={adv.mean():.4f}, median={np.median(adv):.4f}, std={adv.std():.4f}")
        print(f"  Normalized advantage: mean={adv_norm.mean():.4f}, median={np.median(adv_norm):.4f}")

    all_norm_means = [r["advantage_normalized"].mean() for r in results]
    print(f"\n=== Across {len(results)} seeds ===")
    print(f"  Normalized advantage mean: {np.mean(all_norm_means):.4f} "
          f"(range: {np.min(all_norm_means):.4f} to {np.max(all_norm_means):.4f})")

    return results