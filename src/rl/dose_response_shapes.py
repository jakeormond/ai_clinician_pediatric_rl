"""
Dose-response shape analysis for action-sensitivity sweeps.

For each timestep's sweep curve (Q vs. swept dose, from action_sensitivity_sweep),
classifies the shape via Spearman correlation between dose and Q:
    - "increasing": strong positive correlation (Q rises with dose)
    - "decreasing": strong negative correlation (Q falls with dose)
    - "non-monotonic": weak correlation despite a real range (U-shaped,
       inverted-U, or genuinely noisy — these look the same to a single
       correlation coefficient, so example curves are pulled for visual check)
    - "flat": small range relative to the rest of the sweep — nothing
       meaningful to classify (down-weighted rather than misclassified as
       "non-monotonic" alongside genuine U-shapes)

Expects, already in your session:
    generate_model_outputs, action_sensitivity_sweep, normalize_sensitivity
    processors['test'], clinician_actions
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from src.rl.evaluate_terminal_reward_models import generate_model_outputs, \
    build_joint_state_action_occupancy_counts, _bin_index, action_sensitivity_sweep

MONOTONIC_COLORS = {
    "increasing": "#229954",
    "decreasing": "#c0392b",
    "non-monotonic": "#7d3c98",
    "flat": "#95a5a6",
}

CURVATURE_COLORS = {"convex (U-shaped)": "#2471a3", "concave (inverted-U)": "#c0392b"}

def classify_dose_response_shapes(
        q_at_grid, grid,
        rho_threshold=0.8,       # |rho| above this -> monotonic increasing/decreasing
        flat_range_pct=5.0,      # timesteps whose own range is below this % of the
                                  # POPULATION median range are called "flat" rather
                                  # than judged on correlation (which is unstable for
                                  # near-constant curves)
    ):
    """
    Returns a dict with per-timestep: rho, pval, range, and a shape label.
    """
    q = q_at_grid.cpu().numpy() if torch.is_tensor(q_at_grid) else np.asarray(q_at_grid)
    grid_np = grid.cpu().numpy() if torch.is_tensor(grid) else np.asarray(grid)

    n_timesteps = q.shape[0]
    q_range = q.max(axis=1) - q.min(axis=1)
    median_range = np.median(q_range)
    flat_cutoff = median_range * (flat_range_pct / 100.0)

    rhos = np.zeros(n_timesteps)
    pvals = np.zeros(n_timesteps)
    labels = np.empty(n_timesteps, dtype=object)

    for t in range(n_timesteps):
        rho, pval = stats.spearmanr(grid_np, q[t])
        rhos[t] = rho
        pvals[t] = pval

        if q_range[t] < flat_cutoff:
            labels[t] = "flat"
        elif rho >= rho_threshold:
            labels[t] = "increasing"
        elif rho <= -rho_threshold:
            labels[t] = "decreasing"
        else:
            labels[t] = "non-monotonic"

    return {"rho": rhos, "pval": pvals, "range": q_range, "label": labels}


def compute_seed_shape_summaries(
        models, processor, clinician_actions, sweep_dim,
        rho_threshold=0.8, flat_range_pct=5.0,
        dim_label="", condition_label="",
    ):
    """
    Runs the sweep + shape classification for every seed's model. Returns:
      - proportions: dict of {label: array of per-seed proportions}
      - example_curves: dict of {label: list of (grid, q_curve) example
        curves — first qualifying example per seed, for visual spot-checking}
    """
    other_dim = 0 if sweep_dim == 1 else 1
    n_seeds = len(models)

    proportions = {label: [] for label in MONOTONIC_COLORS}
    example_curves = {label: [] for label in MONOTONIC_COLORS}

    for seed_idx, model_entry in enumerate(models):
        t0 = time.time()
        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: running sweep...", flush=True)
        model = model_entry[0] if isinstance(model_entry, (list, tuple)) else model_entry

        sweep = action_sensitivity_sweep(processor, model, clinician_actions, sweep_dim=sweep_dim)
        shapes = classify_dose_response_shapes(
            sweep["q_at_grid"], sweep["grid"], rho_threshold=rho_threshold, flat_range_pct=flat_range_pct
        )

        n_t = len(shapes["label"])
        for label in MONOTONIC_COLORS:
            mask = shapes["label"] == label
            proportions[label].append(mask.sum() / n_t)
            if mask.sum() > 0:
                idx = np.where(mask)[0][0]  # first example for this seed/label
                q_np = sweep["q_at_grid"].cpu().numpy() if torch.is_tensor(sweep["q_at_grid"]) else sweep["q_at_grid"]
                grid_np = sweep["grid"].cpu().numpy() if torch.is_tensor(sweep["grid"]) else sweep["grid"]
                example_curves[label].append((grid_np, q_np[idx]))

        print(f"  [{condition_label} / {dim_label}] seed {seed_idx + 1}/{n_seeds}: done in {time.time() - t0:.1f}s — "
              + ", ".join(f"{label}={np.mean(proportions[label][-1])*100:.0f}%" for label in MONOTONIC_COLORS),
              flush=True)

    return {label: np.array(vals) for label, vals in proportions.items()}, example_curves


def plot_dose_response_shapes(
        models, processor, clinician_actions, sweep_dim,
        dim_label, condition_label,
        rho_threshold=0.8, flat_range_pct=5.0,
        n_example_curves_per_label=4,
        save_path=None,
    ):
    """
    Two-panel figure: (A) stacked bar of shape proportions per seed,
    (B) example curves from each shape category, for visual confirmation
    the classification matches what the curves actually look like.
    """
    proportions, example_curves = compute_seed_shape_summaries(
        models, processor, clinician_actions, sweep_dim,
        rho_threshold=rho_threshold, flat_range_pct=flat_range_pct,
        dim_label=dim_label, condition_label=condition_label,
    )

    fig, (ax_bar, ax_curves) = plt.subplots(1, 2, figsize=(13, 5))

    n_seeds = len(next(iter(proportions.values())))
    bottom = np.zeros(n_seeds)
    for label, color in MONOTONIC_COLORS.items():
        vals = proportions[label] * 100
        ax_bar.bar(np.arange(n_seeds), vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax_bar.set_xticks(np.arange(n_seeds))
    ax_bar.set_xticklabels([f"Seed {i+1}" for i in range(n_seeds)])
    ax_bar.set_ylabel("% of timesteps")
    ax_bar.set_title(f"{condition_label} — {dim_label}\ndose-response shape, per seed", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
    ax_bar.spines[["top", "right"]].set_visible(False)

    for label, color in MONOTONIC_COLORS.items():
        curves = example_curves[label][:n_example_curves_per_label]
        for j, (grid_np, q_curve) in enumerate(curves):
            centered = q_curve - q_curve.mean()
            ax_curves.plot(grid_np, centered, color=color, alpha=0.7, linewidth=1.5,
                           label=label if j == 0 else None)
    ax_curves.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_curves.set_xlabel(f"Swept {dim_label.lower()} dose (normalized)")
    ax_curves.set_ylabel("Q, centered per-curve")
    ax_curves.set_title("Example curves per shape category", fontsize=11, fontweight="bold")
    ax_curves.legend(loc="best", frameon=False, fontsize=9)
    ax_curves.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    print(f"\n{condition_label} — {dim_label} — mean proportions across {n_seeds} seeds:")
    for label in MONOTONIC_COLORS:
        print(f"  {label}: {proportions[label].mean()*100:.1f}%  (per-seed: {np.round(proportions[label]*100, 1)})")

    return fig, proportions, example_curves


def fit_quadratic_shape(q_at_grid, grid, non_monotonic_mask):
    """
    For each timestep flagged non-monotonic, fits Q ~ a*dose^2 + b*dose + c
    and returns curvature (a), vertex location (clipped to NaN if outside
    [0,1] — meaning no genuine interior extremum), and fit R^2 (how well a
    parabola actually describes the curve, vs. it being noisier/higher-order).
    """
    q = q_at_grid.cpu().numpy() if torch.is_tensor(q_at_grid) else np.asarray(q_at_grid)
    grid_np = grid.cpu().numpy() if torch.is_tensor(grid) else np.asarray(grid)
    idxs = np.where(non_monotonic_mask)[0]
 
    a_vals, vertex_vals, r2_vals = [], [], []
    for i in idxs:
        y = q[i]
        coeffs = np.polyfit(grid_np, y, deg=2)
        a, b, c = coeffs
        fitted = np.polyval(coeffs, grid_np)
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
 
        vertex = -b / (2 * a) if abs(a) > 1e-10 else np.nan
        if vertex is not np.nan and not (0.0 <= vertex <= 1.0):
            vertex = np.nan  # not a genuine interior extremum — edge-dominated shape
 
        a_vals.append(a)
        vertex_vals.append(vertex)
        r2_vals.append(r2)
 
    return {
        "timestep_idx": idxs,
        "curvature_a": np.array(a_vals),
        "vertex_location": np.array(vertex_vals),
        "r2": np.array(r2_vals),
    }
 
 
 
def occupancy_support_at_vertex(
        vertex_locations, real_other_dose, test_health_score, sweep_dim,
        joint_counts, severity_edges, fluid_edges, vaso_edges,
    ):
    """
    For each non-monotonic timestep with a genuine interior vertex, looks up
    whether the (severity, fluid, vaso) combination AT the fitted vertex dose
    has real training support — same 3D occupancy histogram as the main
    occupancy-partitioned sensitivity analysis. Returns the raw occupancy
    count at each vertex (NaN where there's no genuine interior vertex).
    """
    valid = ~np.isnan(vertex_locations)
    counts = np.full(len(vertex_locations), np.nan)
 
    severity_bin_full = _bin_index(
        test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score),
        severity_edges,
    )
    real_other_np = real_other_dose.cpu().numpy() if torch.is_tensor(real_other_dose) else np.asarray(real_other_dose)
 
    for i in np.where(valid)[0]:
        sev_bin = severity_bin_full[i]
        if sweep_dim == 1:  # vaso swept -> vertex is a vaso dose, fluid held at real dose
            f_bin = _bin_index(np.array([real_other_np[i]]), fluid_edges)[0]
            v_bin = _bin_index(np.array([vertex_locations[i]]), vaso_edges)[0]
        else:
            f_bin = _bin_index(np.array([vertex_locations[i]]), fluid_edges)[0]
            v_bin = _bin_index(np.array([real_other_np[i]]), vaso_edges)[0]
        counts[i] = joint_counts[sev_bin, f_bin, v_bin]
 
    return counts
 
 

def analyze_non_monotonic_states(
        model, processor, clinician_actions, sweep_dim,
        test_health_score, training_actions, training_health_score,
        mortality=None,
        rho_threshold=0.8, flat_range_pct=5.0,
        n_severity_bins=20, n_action_bins=10, min_count=10,
        label="",
        save_path=None,
    ):
    """
    Full pipeline for ONE model: sweep -> classify shapes -> fit quadratics to
    the non-monotonic subset -> check occupancy support at their vertices ->
    compare severity/dose distributions of non-monotonic vs. monotonic states.
    """
    other_dim = 0 if sweep_dim == 1 else 1
    real_other_dose = clinician_actions[:, other_dim]
 
    sweep = action_sensitivity_sweep(processor, model, clinician_actions, sweep_dim=sweep_dim)
    shapes = classify_dose_response_shapes(sweep["q_at_grid"], sweep["grid"],
                                            rho_threshold=rho_threshold, flat_range_pct=flat_range_pct)
    non_mono_mask = shapes["label"] == "non-monotonic"
    n_non_mono = non_mono_mask.sum()
    print(f"[{label}] {n_non_mono} non-monotonic timesteps out of {len(shapes['label'])} "
          f"({n_non_mono / len(shapes['label']) * 100:.1f}%)")
 
    quad = fit_quadratic_shape(sweep["q_at_grid"], sweep["grid"], non_mono_mask)
 
    convex_mask = quad["curvature_a"] > 0
    concave_mask = quad["curvature_a"] < 0
    interior_vertex_mask = ~np.isnan(quad["vertex_location"])
 
    print(f"[{label}]  of non-monotonic: {convex_mask.mean()*100:.1f}% convex (U-shaped), "
          f"{concave_mask.mean()*100:.1f}% concave (inverted-U)")
    print(f"[{label}]  {interior_vertex_mask.mean()*100:.1f}% have a genuine interior vertex "
          f"(rest are edge-dominated / not a true turning point within [0,1])")
    print(f"[{label}]  mean R^2 of quadratic fit: {quad['r2'].mean():.2f} "
          f"(low = shape is noisier/higher-order than a simple U/inverted-U)")
 
    joint_counts, severity_edges, fluid_edges, vaso_edges = None, None, None, None
    if interior_vertex_mask.sum() > 0:
        joint_counts, severity_edges, fluid_edges, vaso_edges = build_joint_state_action_occupancy_counts(
            training_health_score, training_actions, n_severity_bins=n_severity_bins, n_action_bins=n_action_bins
        )
        vertex_ts_idx = quad["timestep_idx"][interior_vertex_mask]
        counts_at_vertex = occupancy_support_at_vertex(
            quad["vertex_location"][interior_vertex_mask],
            real_other_dose[vertex_ts_idx], test_health_score[vertex_ts_idx], sweep_dim,
            joint_counts, severity_edges, fluid_edges, vaso_edges,
        )
        pct_well_supported = (counts_at_vertex >= min_count).mean()
        print(f"[{label}]  of interior-vertex curves: {pct_well_supported*100:.1f}% have a "
              f"well-supported (n>={min_count}) dose combination AT the fitted extremum "
              f"(low = extrema mostly sit in sparse/extrapolated territory)")
 
    # --- severity comparison: non-monotonic vs. rest ---
    hs = test_health_score.cpu().numpy() if torch.is_tensor(test_health_score) else np.asarray(test_health_score)
    fig, axes = plt.subplots(1, 2 if interior_vertex_mask.sum() > 0 else 1, figsize=(11, 4.5))
    if interior_vertex_mask.sum() == 0:
        axes = [axes]
 
    axes[0].hist(hs[~non_mono_mask], bins=30, alpha=0.5, density=True, label="Monotonic/flat", color="#95a5a6")
    axes[0].hist(hs[non_mono_mask], bins=30, alpha=0.5, density=True, label="Non-monotonic", color="#7d3c98")
    axes[0].set_xlabel("Health score (severity)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{label}: severity distribution", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
 
    if interior_vertex_mask.sum() > 0:
        vertex_vals = quad["vertex_location"][interior_vertex_mask]
        curvature_vals = quad["curvature_a"][interior_vertex_mask]
        convex_interior = curvature_vals > 0
        concave_interior = curvature_vals < 0
 
        axes[1].hist(vertex_vals[convex_interior], bins=20, range=(0, 1),
                     color=CURVATURE_COLORS["convex (U-shaped)"], alpha=0.6,
                     label=f"convex (U-shaped), n={convex_interior.sum()}")
        axes[1].hist(vertex_vals[concave_interior], bins=20, range=(0, 1),
                     color=CURVATURE_COLORS["concave (inverted-U)"], alpha=0.6,
                     label=f"concave (inverted-U), n={concave_interior.sum()}")
        axes[1].set_xlabel("Fitted vertex location (normalized dose)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"{label}: where interior extrema sit", fontsize=10, fontweight="bold")
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].spines[["top", "right"]].set_visible(False)
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")


    summary = {
        "pct_non_monotonic": n_non_mono / len(shapes['label']) * 100,
        "pct_convex": convex_mask.mean() * 100 if len(convex_mask) > 0 else np.nan,
        "pct_concave": concave_mask.mean() * 100 if len(concave_mask) > 0 else np.nan,
        "pct_interior_vertex": interior_vertex_mask.mean() * 100 if len(interior_vertex_mask) > 0 else np.nan,
        "mean_r2": quad["r2"].mean() if len(quad["r2"]) > 0 else np.nan,
        "pct_well_supported": pct_well_supported * 100 if interior_vertex_mask.sum() > 0 else np.nan,
    }
 
    return {
        "shapes": shapes,
        "quad": quad,
        "joint_counts": joint_counts,
        "severity_edges": severity_edges,
        "summary": summary,
    }

