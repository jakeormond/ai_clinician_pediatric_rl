"""
Runs the dose-response shape analysis (analyze_non_monotonic_states, from
analyze_non_monotonic.py) on the final CQL critic's fluid dose-response,
per seed, then cross-references the CONVEX (U-shaped, "worse in the
middle") non-monotonic states against the residual high-dose (>0.9)
states already identified via check_std_dose_cooccurrence.

Purpose: test whether the residual extreme-dose recommendation traces
back to a genuine shape anomaly in the CQL critic itself (inherited from
or introduced during conservative training), rather than being purely an
actor-side (mean-saturation) artifact.
"""

import numpy as np
import torch

CEILING_TOL = 1e-2


def get_cql_critic(model_entry):
    """CQL tuples are (actor, critic, config, hyperparams)."""
    return model_entry[1] if isinstance(model_entry, (list, tuple)) else model_entry


def get_cql_actor(model_entry):
    return model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry


def analyze_shape_dose_overlap(
        model_entry, processor, clinician_actions,
        test_health_score, training_actions, training_health_score,
        dose_threshold=0.9, n_severity_bins=20, n_action_bins=10, min_count=10,
        rho_threshold=0.8, flat_range_pct=5.0,
        label="", save_path=None,
    ):
    """
    Returns the analyze_non_monotonic_states result dict, plus overlap
    statistics between convex-non-monotonic states and high-dose states.
    """
    critic = get_cql_critic(model_entry)
    actor = get_cql_actor(model_entry)

    # 1. shape analysis on the critic
    result = analyze_non_monotonic_states(
        critic, processor, clinician_actions, sweep_dim=0,  # fluid
        test_health_score=test_health_score,
        training_actions=training_actions,
        training_health_score=training_health_score,
        rho_threshold=rho_threshold, flat_range_pct=flat_range_pct,
        n_severity_bins=n_severity_bins, n_action_bins=n_action_bins, min_count=min_count,
        label=label, save_path=save_path,
    )

    shapes = result["shapes"]
    quad = result["quad"]
    n_total = len(shapes["label"])

    convex_mask_local = quad["curvature_a"] > 0          # within the non-monotonic subset
    convex_full_idx = quad["timestep_idx"][convex_mask_local]  # map back to full-array indices

    convex_full_mask = np.zeros(n_total, dtype=bool)
    convex_full_mask[convex_full_idx] = True

    # 2. high-dose states from the actor
    actions = get_actor_actions(model_entry, processor)
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    high_dose_mask = actions_np[:, 0] > dose_threshold

    n_convex = convex_full_mask.sum()
    n_high_dose = high_dose_mask.sum()
    n_both = (convex_full_mask & high_dose_mask).sum()

    print(f"[{label}] Convex (U-shaped) states: {n_convex} ({n_convex/n_total*100:.2f}% of all states)")
    print(f"[{label}] High-dose (>{dose_threshold}) states: {n_high_dose} ({n_high_dose/n_total*100:.2f}% of all states)")
    print(f"[{label}] States that are BOTH: {n_both}")
    if n_convex > 0:
        print(f"[{label}]   Fraction of convex states that are ALSO high-dose: {n_both/n_convex*100:.1f}%")
    if n_high_dose > 0:
        print(f"[{label}]   Fraction of high-dose states that are ALSO convex: {n_both/n_high_dose*100:.1f}%")

    return {
        "shape_result": result,
        "convex_full_mask": convex_full_mask,
        "high_dose_mask": high_dose_mask,
        "n_convex": n_convex, "n_high_dose": n_high_dose, "n_both": n_both,
    }


def run_across_seeds(models_by_seed, processor, clinician_actions,
                      test_health_score, training_actions, training_health_score,
                      save_dir=None, **kwargs):
    results = []
    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"\n=== Seed {seed_idx} ===")
        save_path = f"{save_dir}/shape_dose_overlap_seed{seed_idx}.png" if save_dir else None
        res = analyze_shape_dose_overlap(
            model_entry, processor, clinician_actions,
            test_health_score, training_actions, training_health_score,
            label=f"seed{seed_idx}", save_path=save_path,
            **kwargs,
        )
        results.append(res)
    return results


if __name__ == "__main__":
    clinician_actions_test = processors['test'].rl_dataset["actions"]
    results = run_across_seeds(
        cql_health_score_fixed_alpha_models,
        processors['test'], clinician_actions_test,
        test_health_score, training_actions, training_health_score,
        save_dir=cql_model_dir,
    )