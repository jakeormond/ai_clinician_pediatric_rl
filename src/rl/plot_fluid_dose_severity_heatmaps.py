"""
2D histogram heatmaps of fluid dose vs. severity (health score), faceted
by vasopressor status -- a more honest view than a scatter plot for
checking whether the dose=1.0 spike is genuinely concentrated among
vaso-given/sick patients, or present at meaningful density across the
whole severity range (including high health scores / no vaso).

Uses log-scale color (counts) since dose=1.0 concentration, if real, will
vastly outweigh surrounding cells and wash out on a linear scale.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_fluid_dose_severity_heatmaps(
        f_raw_mean, health_score, clinician_actions,
        vaso_threshold=1e-4, n_severity_bins=40, n_dose_bins=40,
        normalize="severity_bin", save_path=None,
    ):
    """
    normalize:
        "severity_bin" (default) -- within each severity bin (column), show
            the CONDITIONAL distribution of dose (rows sum to 1 per column).
            This answers "given a patient at this severity, how likely are
            they to get each dose level" -- removes the confound of simply
            having more total observations at some severities than others.
        "none" -- raw counts (log scale), the original version; useful for
            seeing absolute density but conflates severity-distribution
            skew with dose-response.
    """
    health_score_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)
    actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    vaso_given = actions_np[:, 1] > vaso_threshold

    dose = 1 / (1 + np.exp(-f_raw_mean))  # sigmoid, dose-space

    hs_edges = np.linspace(health_score_np.min(), health_score_np.max(), n_severity_bins + 1)
    dose_edges = np.linspace(0, 1, n_dose_bins + 1)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.5))
    heatmap_axes = axes[:3]
    line_ax = axes[3]

    groups = [
        ("No vasopressor", ~vaso_given),
        ("Vasopressor given", vaso_given),
        ("All patients", np.ones_like(vaso_given, dtype=bool)),
    ]

    for ax, (label, mask) in zip(heatmap_axes, groups):
        counts, xedges, yedges = np.histogram2d(
            health_score_np[mask], dose[mask], bins=[hs_edges, dose_edges]
        )
        # counts shape: (n_severity_bins, n_dose_bins) -- rows=severity, cols=dose

        if normalize == "severity_bin":
            col_sums = counts.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                plotted = np.where(col_sums > 0, counts / col_sums, 0)
            cbar_label = "P(dose bin | severity bin)"
            # Clip the color scale well below the max: the dose~1.0 row
            # dominates every column, saturating a max-scaled colorbar and
            # hiding all mid-range structure. vmax is capped at a fixed,
            # modest value (or a percentile) so the REST of the distribution
            # becomes visible; the dose~1.0 row will simply appear uniformly
            # "maxed out" rather than showing its own internal gradation --
            # that row is examined separately in the line plot below instead.
            norm = None
            non_extreme_mask = yedges[:-1] < 0.95  # exclude the top dose bin(s) from vmax calc
            vmax = np.percentile(plotted[:, non_extreme_mask], 99) if non_extreme_mask.any() else plotted.max()
            vmax = max(vmax, 1e-3)
            vmin = 0
        else:
            plotted = counts
            cbar_label = "Count (log scale)"
            norm = LogNorm(vmin=1, vmax=max(counts.max(), 1))
            vmin, vmax = None, None

        im = ax.pcolormesh(xedges, yedges, plotted.T, cmap="viridis", norm=norm, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Health score (severity)")
        ax.set_ylabel("Fluid dose (mean action)")
        ax.set_title(f"{label}\n(n={mask.sum()})", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label=cbar_label)

        # overlay the raw per-severity-bin sample size as a thin annotation,
        # so a reader can see where the conditional estimate is well- vs
        # poorly-supported (e.g. very few patients in the sickest bins)
        bin_n = counts.sum(axis=1)
        ax2 = ax.twinx()
        ax2.plot((xedges[:-1] + xedges[1:]) / 2, bin_n, color="white", linewidth=1, alpha=0.6)
        ax2.set_yscale("log")
        ax2.set_ylabel("n per severity bin (log)", color="grey", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7, colors="grey")

    title = ("Fluid dose vs. severity, by vasopressor status\n"
             "(conditional on severity bin -- corrects for uneven severity distribution)"
             if normalize == "severity_bin" else
             "Fluid dose vs. severity, by vasopressor status (raw log-scale density)")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)

    # --- fourth panel: P(dose >= 0.9) vs severity, isolated and directly
    # comparable across vaso/no-vaso, independent of colorbar-scaling issues ---
    high_dose_threshold = 0.9
    for label, mask, color in [("No vasopressor", ~vaso_given, "#2471a3"),
                                ("Vasopressor given", vaso_given, "#c0392b"),
                                ("All patients", np.ones_like(vaso_given, dtype=bool), "grey")]:
        bin_idx = np.digitize(health_score_np[mask], hs_edges[1:-1])
        bin_idx = np.clip(bin_idx, 0, n_severity_bins - 1)
        prob_high = np.full(n_severity_bins, np.nan)
        n_per_bin = np.zeros(n_severity_bins)
        for b in range(n_severity_bins):
            in_bin = bin_idx == b
            n_per_bin[b] = in_bin.sum()
            if in_bin.sum() > 0:
                prob_high[b] = (dose[mask][in_bin] >= high_dose_threshold).mean()
        bin_centers = (hs_edges[:-1] + hs_edges[1:]) / 2
        line_ax.plot(bin_centers, prob_high, color=color, label=label, linewidth=2,
                     alpha=1.0 if label != "All patients" else 0.5,
                     linestyle="--" if label == "All patients" else "-")

    line_ax.set_xlabel("Health score (severity)")
    line_ax.set_ylabel(f"P(fluid dose \u2265 {high_dose_threshold})")
    line_ax.set_title(f"(d) Prob. of near-maximal dose vs. severity", fontsize=11, fontweight="bold")
    line_ax.legend(frameon=False, fontsize=9)
    line_ax.set_ylim(bottom=0)
    line_ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    high_dose = dose > 0.9
    healthy_no_vaso = (health_score_np > 70) & (~vaso_given)
    print(f"\nFraction of ALL states with dose>0.9: {high_dose.mean():.4f}")
    print(f"Fraction of states with dose>0.9 that are ALSO (health_score>70 AND no vaso): "
          f"{(high_dose & healthy_no_vaso).sum() / high_dose.sum():.4f}")
    print(f"  (i.e. this fraction of the high-dose spike is NOT explained by sickness+vaso)")
    print(f"For comparison, (health_score>70 AND no vaso) as a fraction of ALL states: "
          f"{healthy_no_vaso.mean():.4f}")
    print(f"  (if the two numbers above are similar, the spike is NOT concentrated away from "
          f"this group -- i.e. sickness/vaso does NOT explain the spike)")

    return fig


if __name__ == "__main__":
    clinician_actions_test = processors['test'].rl_dataset["actions"]
    fig = plot_fluid_dose_severity_heatmaps(
        f_raw_mean, test_health_score, clinician_actions_test,
        save_path="fluid_dose_severity_heatmaps.png",
    )