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
import matplotlib.pyplot as plt

from src.rl.dose_response_shapes import (
    action_sensitivity_sweep, classify_dose_response_shapes,
    fit_quadratic_shape, analyze_non_monotonic_states,
)
from src.rl.action_distribution_sweep import get_actor_actions
from src.rl.dose_response_shapes import (
    build_joint_state_action_occupancy_counts, _bin_index,
)

CEILING_TOL = 1e-2

CATEGORY_COLORS = {
    "Increasing": "#229954",
    "Decreasing": "#c0392b",
    "Concave (inverted-U)": "#2471a3",
    "Convex (U-shaped)": "#7d3c98",
    "Flat": "#95a5a6",
}


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

    result = analyze_non_monotonic_states(
        critic, processor, clinician_actions, sweep_dim=0,
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

    convex_mask_local = quad["curvature_a"] > 0
    convex_full_idx = quad["timestep_idx"][convex_mask_local]

    convex_full_mask = np.zeros(n_total, dtype=bool)
    convex_full_mask[convex_full_idx] = True

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


def analyze_monotonic_increasing_dose_overlap(
        model_entry, processor, clinician_actions,
        test_health_score, training_actions, training_health_score,
        dose_threshold=0.9, n_severity_bins=20, n_action_bins=10, min_count=10,
        rho_threshold=0.8, flat_range_pct=5.0,
        label="", save_path=None,
    ):
    """
    Companion check to analyze_shape_dose_overlap: tests whether the
    residual high-dose (>0.9) states are better explained by a
    straightforwardly MONOTONICALLY INCREASING Q(dose) relationship
    rather than by non-monotonic convex (U-shaped) structure.
    """
    critic = get_cql_critic(model_entry)

    result = analyze_non_monotonic_states(
        critic, processor, clinician_actions, sweep_dim=0,
        test_health_score=test_health_score,
        training_actions=training_actions,
        training_health_score=training_health_score,
        rho_threshold=rho_threshold, flat_range_pct=flat_range_pct,
        n_severity_bins=n_severity_bins, n_action_bins=n_action_bins, min_count=min_count,
        label=label, save_path=None,
    )

    shapes = result["shapes"]
    n_total = len(shapes["label"])
    increasing_mask = shapes["label"] == "increasing"

    actions = get_actor_actions(model_entry, processor)
    actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
    high_dose_mask = actions_np[:, 0] > dose_threshold

    n_increasing = increasing_mask.sum()
    n_high_dose = high_dose_mask.sum()
    n_both = (increasing_mask & high_dose_mask).sum()

    print(f"[{label}] Monotonically increasing states: {n_increasing} ({n_increasing/n_total*100:.2f}% of all states)")
    print(f"[{label}] High-dose (>{dose_threshold}) states: {n_high_dose} ({n_high_dose/n_total*100:.2f}% of all states)")
    print(f"[{label}] States that are BOTH: {n_both}")
    if n_increasing > 0:
        print(f"[{label}]   Fraction of increasing states that are ALSO high-dose: {n_both/n_increasing*100:.1f}%")
    if n_high_dose > 0:
        print(f"[{label}]   Fraction of high-dose states that are ALSO increasing: {n_both/n_high_dose*100:.1f}%")

    return {
        "shapes": shapes,
        "increasing_mask": increasing_mask,
        "high_dose_mask": high_dose_mask,
        "n_increasing": n_increasing, "n_high_dose": n_high_dose, "n_both": n_both,
    }


def run_monotonic_check_across_seeds(models_by_seed, processor, clinician_actions,
                                       test_health_score, training_actions, training_health_score,
                                       **kwargs):
    results = []
    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"\n=== Seed {seed_idx} (monotonic-increasing check) ===")
        res = analyze_monotonic_increasing_dose_overlap(
            model_entry, processor, clinician_actions,
            test_health_score, training_actions, training_health_score,
            label=f"seed{seed_idx}",
            **kwargs,
        )
        results.append(res)
    return results


def check_boundary_occupancy_for_increasing_states(
        model_entry, processor, clinician_actions,
        test_health_score, training_actions, training_health_score,
        n_severity_bins=50, n_action_bins=10, min_count=10,
        rho_threshold=0.8, flat_range_pct=5.0,
        label="",
    ):
    """
    For states whose fluid dose-response is classified as monotonically
    increasing, checks whether the dose combination AT THE SWEPT BOUNDARY
    (fluid=1.0, real clinician vaso dose, this state's own severity) was
    actually observed >= min_count times in the training data.
    """
    critic = get_cql_critic(model_entry)

    sweep = action_sensitivity_sweep(processor, critic, clinician_actions, sweep_dim=0)
    shapes = classify_dose_response_shapes(
        sweep["q_at_grid"], sweep["grid"], rho_threshold=rho_threshold, flat_range_pct=flat_range_pct
    )
    increasing_mask = shapes["label"] == "increasing"

    joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
    )

    hs_np = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    real_vaso = clinician_actions_np[:, 1]

    grid_np = sweep["grid"].cpu().numpy() if torch.is_tensor(sweep["grid"]) else np.asarray(sweep["grid"])
    boundary_fluid_val = grid_np[-1]

    sev_bin = _bin_index(hs_np, severity_edges)
    vaso_bin = _bin_index(real_vaso, vaso_edges)
    fluid_bin_boundary = _bin_index(np.array([boundary_fluid_val]), fluid_edges)[0]

    counts_at_boundary = joint_counts[sev_bin, fluid_bin_boundary, vaso_bin]
    well_supported_boundary = counts_at_boundary >= min_count

    n_increasing = increasing_mask.sum()
    if n_increasing > 0:
        frac_well_supported = well_supported_boundary[increasing_mask].mean()
        median_count = np.median(counts_at_boundary[increasing_mask])
    else:
        frac_well_supported = np.nan
        median_count = np.nan

    print(f"[{label}] Monotonically increasing states: {n_increasing} ({n_increasing/len(shapes['label'])*100:.2f}%)")
    print(f"[{label}]   Of these, {frac_well_supported*100:.1f}% have a well-supported (n>={min_count}) "
          f"dose combination AT the boundary (fluid={boundary_fluid_val:.3f})")
    print(f"[{label}]   Median training-data count at boundary, for these states: {median_count:.0f}")

    return {
        "increasing_mask": increasing_mask,
        "counts_at_boundary": counts_at_boundary,
        "well_supported_boundary": well_supported_boundary,
        "frac_well_supported": frac_well_supported,
    }


def run_boundary_occupancy_check_across_seeds(models_by_seed, processor, clinician_actions,
                                                test_health_score, training_actions, training_health_score,
                                                **kwargs):
    results = []
    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"\n=== Seed {seed_idx} (boundary occupancy check) ===")
        res = check_boundary_occupancy_for_increasing_states(
            model_entry, processor, clinician_actions,
            test_health_score, training_actions, training_health_score,
            label=f"seed{seed_idx}",
            **kwargs,
        )
        results.append(res)
    return results


def compute_full_shape_breakdown(model_entry, test_health_score,
                                   rho_threshold=0.8, flat_range_pct=5.0, processor=None, clinician_actions=None):
    """
    Runs the sweep + shape classification + curvature fit for one model,
    returning per-timestep category labels and the corresponding health
    scores -- the raw material for both panels of the figure.
    """
    critic = get_cql_critic(model_entry)
    sweep = action_sensitivity_sweep(processor, critic, clinician_actions, sweep_dim=0)
    shapes = classify_dose_response_shapes(
        sweep["q_at_grid"], sweep["grid"], rho_threshold=rho_threshold, flat_range_pct=flat_range_pct
    )
    non_mono_mask = shapes["label"] == "non-monotonic"
    quad = fit_quadratic_shape(sweep["q_at_grid"], sweep["grid"], non_mono_mask)

    n_total = len(shapes["label"])
    category = np.full(n_total, "Flat", dtype=object)
    category[shapes["label"] == "increasing"] = "Increasing"
    category[shapes["label"] == "decreasing"] = "Decreasing"
    convex_idx = quad["timestep_idx"][quad["curvature_a"] > 0]
    concave_idx = quad["timestep_idx"][quad["curvature_a"] < 0]
    category[convex_idx] = "Convex (U-shaped)"
    category[concave_idx] = "Concave (inverted-U)"

    hs_np = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
    return category, hs_np


def compute_shape_breakdown_data(
        models_by_seed, processor, clinician_actions, test_health_score,
        rho_threshold=0.8, flat_range_pct=5.0,
    ):
    """
    The expensive part (~7 min): runs the sweep + shape classification +
    curvature fit for every seed. Returns per_seed_category, per_seed_hs.
    """
    per_seed_category = []
    per_seed_hs = []
    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"Computing shape breakdown for seed {seed_idx}...", flush=True)
        category, hs_np = compute_full_shape_breakdown(
            model_entry, test_health_score, rho_threshold=rho_threshold, flat_range_pct=flat_range_pct,
            processor=processor, clinician_actions=clinician_actions,
        )
        per_seed_category.append(category)
        per_seed_hs.append(hs_np)
    return per_seed_category, per_seed_hs


def plot_shape_breakdown_figure(per_seed_category, per_seed_hs, save_path=None):
    """
    The cheap part: builds the two-panel figure from ALREADY-COMPUTED
    per_seed_category / per_seed_hs.
      (a) stacked bar of shape-category proportions, one bar per seed
      (b) box plot of health score (severity) by shape category, pooled
          across all 5 seeds
    """
    categories_order = ["Increasing", "Decreasing", "Concave (inverted-U)", "Convex (U-shaped)", "Flat"]
    n_seeds = len(per_seed_category)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    bottom = np.zeros(n_seeds)
    for cat in categories_order:
        pct_per_seed = np.array([(per_seed_category[s] == cat).mean() * 100 for s in range(n_seeds)])
        ax1.bar(np.arange(n_seeds), pct_per_seed, bottom=bottom, color=CATEGORY_COLORS[cat], label=cat)
        bottom += pct_per_seed
    ax1.set_xticks(np.arange(n_seeds))
    ax1.set_xticklabels([f"Seed {i}" for i in range(n_seeds)])
    ax1.set_ylabel("% of timesteps")
    ax1.set_title("(a) Fluid dose-response shape, per seed", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    pooled_category = np.concatenate(per_seed_category)
    pooled_hs = np.concatenate(per_seed_hs)

    box_data = [pooled_hs[pooled_category == cat] for cat in categories_order]
    box_colors = [CATEGORY_COLORS[cat] for cat in categories_order]

    bp = ax2.boxplot(box_data, showfliers=False, patch_artist=True,
                      medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xticks(range(1, len(categories_order) + 1))
    ax2.set_xticklabels(categories_order, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Health score (severity)")
    ax2.set_title("(b) Severity by dose-response shape\n(pooled across 5 seeds)", fontsize=11, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fluid dose-response shape composition and its relationship to severity",
                 fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)

    print("\n=== Pooled category counts (all 5 seeds combined) ===")
    for cat in categories_order:
        n = (pooled_category == cat).sum()
        print(f"  {cat}: n={n} ({n/len(pooled_category)*100:.2f}%), "
              f"median health score={np.median(pooled_hs[pooled_category == cat]):.1f}")

    return fig


def compute_extended_shape_data(
        models_by_seed, processor, clinician_actions,
        test_health_score, training_actions, training_health_score,
        rho_threshold=0.8, flat_range_pct=5.0,
        n_severity_bins=20, n_action_bins=10, min_count=10,
    ):
    """
    Extended version of compute_shape_breakdown_data: captures everything
    needed for the follow-up checks (occupancy at real actions by
    category, R^2/vertex-occupancy by curvature sign, and flat-state
    characterization) in ONE pass per seed.
    """
    joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
    )

    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    hs_np_shared = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
    sev_bin_shared = _bin_index(hs_np_shared, severity_edges)
    real_fluid_bin = _bin_index(clinician_actions_np[:, 0], fluid_edges)
    real_vaso_bin = _bin_index(clinician_actions_np[:, 1], vaso_edges)
    occupancy_at_real_action = joint_counts[sev_bin_shared, real_fluid_bin, real_vaso_bin]

    per_seed_data = []
    for seed_idx, model_entry in enumerate(models_by_seed):
        print(f"Computing extended shape data for seed {seed_idx}...", flush=True)
        critic = get_cql_critic(model_entry)

        sweep = action_sensitivity_sweep(processor, critic, clinician_actions, sweep_dim=0)
        shapes = classify_dose_response_shapes(
            sweep["q_at_grid"], sweep["grid"], rho_threshold=rho_threshold, flat_range_pct=flat_range_pct
        )
        non_mono_mask = shapes["label"] == "non-monotonic"
        quad = fit_quadratic_shape(sweep["q_at_grid"], sweep["grid"], non_mono_mask)

        n_total = len(shapes["label"])
        category = np.full(n_total, "Flat", dtype=object)
        category[shapes["label"] == "increasing"] = "Increasing"
        category[shapes["label"] == "decreasing"] = "Decreasing"
        convex_local = quad["curvature_a"] > 0
        concave_local = quad["curvature_a"] < 0
        category[quad["timestep_idx"][convex_local]] = "Convex (U-shaped)"
        category[quad["timestep_idx"][concave_local]] = "Concave (inverted-U)"

        r2_full = np.full(n_total, np.nan)
        vertex_full = np.full(n_total, np.nan)
        curvature_full = np.full(n_total, np.nan)
        r2_full[quad["timestep_idx"]] = quad["r2"]
        vertex_full[quad["timestep_idx"]] = quad["vertex_location"]
        curvature_full[quad["timestep_idx"]] = quad["curvature_a"]

        # rho (Spearman correlation) is defined for EVERY state (it's what
        # classify_dose_response_shapes uses to decide increasing/decreasing/
        # flat/non-monotonic in the first place), unlike R^2 which only
        # exists for the non-monotonic subset -- so this is a full-length
        # array, no index-mapping needed. NOTE: assumes shapes["rho"] is the
        # correct key -- check against your actual classify_dose_response_shapes
        # return dict and adjust the key name below if it differs.
        rho_full = shapes["rho"] if "rho" in shapes else np.full(n_total, np.nan)

        occupancy_count_vertex = np.full(n_total, np.nan)
        has_interior_vertex = ~np.isnan(vertex_full)
        if has_interior_vertex.sum() > 0:
            idx_with_vertex = np.where(has_interior_vertex)[0]
            v_bin = _bin_index(vertex_full[idx_with_vertex], fluid_edges)
            s_bin = sev_bin_shared[idx_with_vertex]
            va_bin = real_vaso_bin[idx_with_vertex]
            occupancy_count_vertex[idx_with_vertex] = joint_counts[s_bin, v_bin, va_bin]

        per_seed_data.append({
            "category": category,
            "health_score": hs_np_shared,
            "r2_full": r2_full,
            "rho_full": rho_full,
            "vertex_location_full": vertex_full,
            "curvature_a_full": curvature_full,
            "occupancy_count_real_action": occupancy_at_real_action,
            "occupancy_count_vertex": occupancy_count_vertex,
            "clinician_fluid_dose": clinician_actions_np[:, 0],
        })

    return per_seed_data


def check_occupancy_by_category(per_seed_data):
    """Check 1: is Concave better-supported (higher training occupancy at
    the real clinician action) than Increasing/Decreasing?"""
    categories_order = ["Increasing", "Decreasing", "Concave (inverted-U)", "Convex (U-shaped)", "Flat"]
    pooled_category = np.concatenate([d["category"] for d in per_seed_data])
    pooled_occ = np.concatenate([d["occupancy_count_real_action"] for d in per_seed_data])

    print("=== Occupancy support at REAL clinician action, by category ===")
    for cat in categories_order:
        vals = pooled_occ[pooled_category == cat]
        if len(vals) > 0:
            print(f"  {cat}: n={len(vals)}, median count={np.median(vals):.0f}, "
                  f"mean count={vals.mean():.1f}, %>=10={100*(vals>=10).mean():.1f}%")


def check_r2_and_vertex_occupancy_by_curvature(per_seed_data):
    """Check 2: are Convex fits noisier / less-supported than Concave fits?"""
    pooled_curvature = np.concatenate([d["curvature_a_full"] for d in per_seed_data])
    pooled_r2 = np.concatenate([d["r2_full"] for d in per_seed_data])
    pooled_vertex_occ = np.concatenate([d["occupancy_count_vertex"] for d in per_seed_data])

    convex_mask = pooled_curvature > 0
    concave_mask = pooled_curvature < 0

    print("=== R^2 and vertex occupancy, by curvature sign ===")
    for label, mask in [("Convex (U-shaped)", convex_mask), ("Concave (inverted-U)", concave_mask)]:
        r2_vals = pooled_r2[mask]
        occ_vals = pooled_vertex_occ[mask]
        occ_vals_valid = occ_vals[~np.isnan(occ_vals)]
        print(f"  {label}: n={mask.sum()}, median R^2={np.median(r2_vals):.3f}, "
              f"median vertex occupancy count={np.median(occ_vals_valid):.0f} "
              f"(valid interior vertex for {len(occ_vals_valid)}/{mask.sum()} states, "
              f"{100*len(occ_vals_valid)/mask.sum():.1f}%)")


def check_flat_states(per_seed_data, zero_threshold=1e-4):
    """Check 3: are Flat states healthy patients who never received fluid?"""
    pooled_category = np.concatenate([d["category"] for d in per_seed_data])
    pooled_hs = np.concatenate([d["health_score"] for d in per_seed_data])
    pooled_dose = np.concatenate([d["clinician_fluid_dose"] for d in per_seed_data])

    flat_mask = pooled_category == "Flat"
    non_flat_mask = ~flat_mask

    print("=== Flat-state characterization ===")
    print(f"  Flat states: n={flat_mask.sum()}")
    if flat_mask.sum() > 0:
        print(f"    Health score: median={np.median(pooled_hs[flat_mask]):.1f} "
              f"(vs. non-flat median={np.median(pooled_hs[non_flat_mask]):.1f})")
        print(f"    Real clinician fluid fire rate: "
              f"{100*(pooled_dose[flat_mask] > zero_threshold).mean():.1f}% "
              f"(vs. non-flat fire rate={100*(pooled_dose[non_flat_mask] > zero_threshold).mean():.1f}%)")


def check_all_categories_inferred_point_occupancy(
        per_seed_data, training_actions, training_health_score,
        test_health_score, clinician_actions,
        n_severity_bins=20, n_action_bins=10, min_count=10,
    ):
    """
    Fair, corrected version of the occupancy-by-category comparison: for
    EVERY category, checks occupancy support at the point the CRITIC
    actually implies is relevant, not the real clinician action.

      Increasing -> occupancy at the swept BOUNDARY (fluid=1), real vaso
      Decreasing -> occupancy at the swept BOUNDARY (fluid=0), real vaso
      Concave/Convex -> occupancy at the fitted VERTEX
    """
    joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
    )

    hs_np = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    sev_bin = _bin_index(hs_np, severity_edges)
    real_vaso_bin = _bin_index(clinician_actions_np[:, 1], vaso_edges)

    fluid_bin_high = _bin_index(np.array([1.0]), fluid_edges)[0]
    fluid_bin_low = _bin_index(np.array([0.0]), fluid_edges)[0]

    all_increasing_counts, all_decreasing_counts = [], []
    all_concave_vertex_counts, all_convex_vertex_counts = [], []

    for d in per_seed_data:
        category = d["category"]

        inc_mask = category == "Increasing"
        dec_mask = category == "Decreasing"
        conc_mask = category == "Concave (inverted-U)"
        conv_mask = category == "Convex (U-shaped)"

        all_increasing_counts.append(joint_counts[sev_bin[inc_mask], fluid_bin_high, real_vaso_bin[inc_mask]])
        all_decreasing_counts.append(joint_counts[sev_bin[dec_mask], fluid_bin_low, real_vaso_bin[dec_mask]])

        vertex_counts = d["occupancy_count_vertex"]
        conc_vertex = vertex_counts[conc_mask]
        conv_vertex = vertex_counts[conv_mask]
        all_concave_vertex_counts.append(conc_vertex[~np.isnan(conc_vertex)])
        all_convex_vertex_counts.append(conv_vertex[~np.isnan(conv_vertex)])

    increasing_counts = np.concatenate(all_increasing_counts)
    decreasing_counts = np.concatenate(all_decreasing_counts)
    concave_vertex_counts = np.concatenate(all_concave_vertex_counts)
    convex_vertex_counts = np.concatenate(all_convex_vertex_counts)

    print("=== Occupancy support at the CRITIC-IMPLIED point of interest, by category ===")
    print("(Increasing/Decreasing: at the swept boundary. Concave/Convex: at the fitted vertex.)")
    for label, counts in [
        ("Increasing (at fluid=1 boundary)", increasing_counts),
        ("Decreasing (at fluid=0 boundary)", decreasing_counts),
        ("Concave (at vertex)", concave_vertex_counts),
        ("Convex (at vertex)", convex_vertex_counts),
    ]:
        if len(counts) > 0:
            print(f"  {label}: n={len(counts)}, median count={np.median(counts):.0f}, "
                  f"mean count={counts.mean():.1f}, %>={min_count}={100*(counts>=min_count).mean():.1f}%")
        else:
            print(f"  {label}: n=0 (no states)")


def lightweight_dose_sweep(critic, states, real_actions, sweep_dim=0, n_grid=20,
                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Computes Q across a dose grid for a SMALL, specific set of states --
    much cheaper than the full-dataset action_sensitivity_sweep, since
    it's meant for pulling a handful of example curves for a figure.

    Returns (grid, q_at_grid): grid is (n_grid,), q_at_grid is
    (n_examples, n_grid).
    """
    critic.eval()
    states = states.to(device)
    real_actions = real_actions.to(device)
    grid = torch.linspace(0, 1, n_grid)
    q_at_grid = torch.zeros(states.shape[0], n_grid)

    with torch.no_grad():
        for i, val in enumerate(grid):
            actions_i = real_actions.clone()
            actions_i[:, sweep_dim] = val
            q1, q2 = critic(states, actions_i)
            q = ((q1 + q2) / 2.0).squeeze(-1)
            q_at_grid[:, i] = q.cpu()

    return grid.numpy(), q_at_grid.numpy()


def _order_by_selection(values, selection):
    """Shared ranking logic for both r2_selection and rho_selection."""
    if selection == "max":
        return np.argsort(-values)
    elif selection in ("median", "mean"):
        target = np.median(values) if selection == "median" else np.mean(values)
        return np.argsort(np.abs(values - target))
    else:
        raise ValueError(f"Unknown selection='{selection}', use 'max', 'median', or 'mean'")


def select_example_states(per_seed_data, seed_idx, n_per_category=3, seed=0,
                            r2_selection="max", rho_selection="max"):
    """
    Picks representative example state indices per category, from one
    seed's classification results:
      - Concave/Convex: selected by R^2 (quadratic fit quality), per
        r2_selection ("max" = cleanest fit, "median"/"mean" = typical).
      - Increasing/Decreasing: selected by |rho| (Spearman correlation
        strength -- how cleanly monotonic the trend is), per
        rho_selection, using the same "max"/"median"/"mean" semantics.
      - Flat: random sample -- no analogous "strength" metric applies to
        a flat classification (it's defined by total variation being
        small, not by a correlation/fit-quality score), so this category
        is unaffected by either selection parameter and always draws
        randomly.
    """
    data = per_seed_data[seed_idx]
    category = data["category"]
    r2 = data["r2_full"]
    rho = data.get("rho_full")
    rng = np.random.default_rng(seed)

    examples = {}
    for cat in ["Concave (inverted-U)", "Convex (U-shaped)"]:
        idx = np.where(category == cat)[0]
        order = _order_by_selection(r2[idx], r2_selection)
        examples[cat] = idx[order][:n_per_category]

    if rho is not None:
        for cat in ["Increasing", "Decreasing"]:
            idx = np.where(category == cat)[0]
            order = _order_by_selection(np.abs(rho[idx]), rho_selection)
            examples[cat] = idx[order][:n_per_category]
    else:
        # rho_full not present (e.g. data computed before this field was
        # added) -- fall back to random, same as Flat, and say so once.
        print("Note: 'rho_full' not found in per_seed_data -- Increasing/Decreasing "
              "will be selected randomly. Re-run compute_extended_shape_data to enable rho-based selection.")
        for cat in ["Increasing", "Decreasing"]:
            idx = np.where(category == cat)[0]
            n = min(n_per_category, len(idx))
            examples[cat] = rng.choice(idx, size=n, replace=False) if n > 0 else np.array([], dtype=int)

    for cat in ["Flat"]:
        idx = np.where(category == cat)[0]
        n = min(n_per_category, len(idx))
        examples[cat] = rng.choice(idx, size=n, replace=False) if n > 0 else np.array([], dtype=int)

    return examples


def plot_example_dose_response_curves(
        model_entry, processor, clinician_actions, per_seed_data, seed_idx,
        training_actions, training_health_score,
        sweep_dim=0, example_seed=0, y_padding_frac=0.1,
        n_severity_bins=20, n_action_bins=10, min_count=10,
        r2_selection="max", rho_selection="max", save_path=None,
    ):
    """
    Pulls ONE example Q(fluid dose) curve per shape category and plots it,
    one subplot per category, with the y-axis using a SHARED width across
    all 5 panels (the widest range needed among the 5 example curves,
    centered per-panel on that curve's own midpoint) so slopes are
    directly comparable across categories. Markers are placed directly ON
    the curve (via interpolation) at each WELL-SUPPORTED TRAINING-DATA
    fluid dose bin -- the actual fluid doses given (at this example's own
    severity bin and real vaso-fired status) to comparable patients in
    the training set, not just this one state's own single real dose.
    """
    critic = get_cql_critic(model_entry)
    examples = select_example_states(per_seed_data, seed_idx, n_per_category=1, seed=example_seed,
                                      r2_selection=r2_selection, rho_selection=rho_selection)
    r2_full = per_seed_data[seed_idx]["r2_full"]
    rho_full = per_seed_data[seed_idx].get("rho_full")

    states_all = processor.rl_dataset["states"]
    clinician_actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    health_score = per_seed_data[seed_idx]["health_score"]

    joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
        training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
    )
    fluid_edges_clipped = np.clip(fluid_edges, 0.0, 1.0)
    fluid_bin_centers = (fluid_edges_clipped[:-1] + fluid_edges_clipped[1:]) / 2

    # Which vaso bin index corresponds to EXACTLY zero (not-fired), found the
    # same reliable way fluid boundaries are found elsewhere in this file
    # (_bin_index on the literal value), rather than via a bin-CENTER
    # threshold. A center-threshold check (e.g. "bin center > 1e-4") is
    # WRONG here: with n_action_bins equal-width bins spanning [0,1], the
    # very first bin (containing true zero) has a center at roughly
    # 1/(2*n_action_bins), which is > 1e-4 -- so a center-threshold
    # classifies EVERY bin, including the zero bin, as "fired", leaving the
    # "not fired" mask completely empty for the ~92% of states with real
    # vaso dose exactly 0. This was the bug behind "no well-supported bins
    # found" appearing for every single example.
    n_vaso_bins = len(vaso_edges) - 1
    zero_vaso_bin = _bin_index(np.array([0.0]), vaso_edges)[0]
    all_vaso_bin_idx = np.arange(n_vaso_bins)
    not_fired_bin_mask = all_vaso_bin_idx == zero_vaso_bin
    fired_bin_mask = ~not_fired_bin_mask

    categories_order = ["Increasing", "Decreasing", "Concave (inverted-U)", "Convex (U-shaped)", "Flat"]

    curves_by_cat = {}
    for cat in categories_order:
        idx = examples[cat]
        if len(idx) == 0:
            continue
        state_idx = idx[0]
        states_tensor = torch.as_tensor(states_all[state_idx:state_idx+1], dtype=torch.float32)
        real_actions_tensor = torch.as_tensor(clinician_actions_np[state_idx:state_idx+1], dtype=torch.float32)
        grid, q_at_grid = lightweight_dose_sweep(critic, states_tensor, real_actions_tensor, sweep_dim=sweep_dim)
        curves_by_cat[cat] = (state_idx, grid, q_at_grid[0])

    max_range = max((curve.max() - curve.min()) for _, _, curve in curves_by_cat.values())
    half_width = (max_range * (1 + y_padding_frac)) / 2

    training_hs_np = training_health_score.cpu().numpy() if torch.is_tensor(training_health_score) else np.asarray(training_health_score)
    training_actions_np = training_actions.cpu().numpy() if torch.is_tensor(training_actions) else np.asarray(training_actions)
    training_sev_bin_all = _bin_index(training_hs_np, severity_edges)
    training_vaso_fired_all = training_actions_np[:, 1] > 1e-4

    fig, axes = plt.subplots(1, 5, figsize=(23, 4.5))

    for ax, cat in zip(axes, categories_order):
        if cat not in curves_by_cat:
            ax.set_title(f"{cat}\n(no examples)", fontsize=10)
            ax.axis("off")
            continue

        state_idx, grid, curve = curves_by_cat[cat]
        color = CATEGORY_COLORS[cat]

        sev_bin = _bin_index(np.array([health_score[state_idx]]), severity_edges)[0]
        real_vaso_dose = clinician_actions_np[state_idx, 1]
        vaso_fired = real_vaso_dose > 1e-4

        # Full distribution of REAL training-data fluid doses at this state's
        # own severity bin and vaso-fired status -- shown as a histogram on a
        # secondary axis, BEHIND the Q curve, so the reader can directly see
        # whether the curve's shape aligns with where the training data's
        # mass actually sits, rather than a handful of sampled points or a
        # coarse threshold check.
        matching_mask = (training_sev_bin_all == sev_bin) & (training_vaso_fired_all == vaso_fired)
        matching_fluid_doses = training_actions_np[matching_mask, 0]

        ax2 = ax.twinx()
        if len(matching_fluid_doses) > 0:
            ax2.hist(matching_fluid_doses, bins=30, range=(0, 1), color=color, alpha=0.25, zorder=1,
                     label=f"Training dose density (n={len(matching_fluid_doses)})")
        else:
            print(f"[{cat}, state {state_idx}] No matching training states found at this severity/vaso combination.")
        ax2.set_ylabel("Training count", fontsize=8, color="grey")
        ax2.tick_params(axis="y", labelsize=7, colors="grey")
        ax2.spines[["top"]].set_visible(False)

        # The "curve" is not a fit -- it's the literal Q values computed at
        # each of the 20 swept dose grid points, connected by straight line
        # segments. Marking each grid point explicitly makes this clear
        # rather than implying a smooth, continuously-evaluated function.
        ax.plot(grid, curve, color=color, linewidth=1.8, zorder=5)
        ax.scatter(grid, curve, color=color, s=28, marker="D", edgecolors="black",
                   linewidths=0.5, zorder=6, label="Swept dose grid points" if cat == "Increasing" else None)
        ax.set_zorder(ax2.get_zorder() + 1)   # keep the Q curve's axis (and its interactivity) drawn on top
        ax.patch.set_visible(False)           # let the histogram show through the now-transparent curve axis background

        y_mid = (curve.min() + curve.max()) / 2
        ax.set_ylim(y_mid - half_width, y_mid + half_width)

        ax.set_xlabel("Fluid dose")
        if cat == "Increasing":
            ax.set_ylabel("Q value")
        title = f"{cat}\n(HS={health_score[state_idx]:.0f})"
        if not np.isnan(r2_full[state_idx]):
            title += f", R\u00b2={r2_full[state_idx]:.2f}"
        if rho_full is not None and not np.isnan(rho_full[state_idx]):
            title += f", \u03c1={rho_full[state_idx]:.4f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines[["top"]].set_visible(False)

    fig.suptitle("Example fluid dose-response curves by shape category\n"
                 "(markers: well-supported training-data dose bins at matching severity/vaso)",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.close(fig)
    return fig