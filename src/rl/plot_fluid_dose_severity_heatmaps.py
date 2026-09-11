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
        save_path=None,
    ):
    health_score_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)
    actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    vaso_given = actions_np[:, 1] > vaso_threshold

    dose = 1 / (1 + np.exp(-f_raw_mean))  # sigmoid, dose-space

    hs_edges = np.linspace(health_score_np.min(), health_score_np.max(), n_severity_bins + 1)
    dose_edges = np.linspace(0, 1, n_dose_bins + 1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    groups = [
        ("No vasopressor", ~vaso_given),
        ("Vasopressor given", vaso_given),
        ("All patients", np.ones_like(vaso_given, dtype=bool)),
    ]

    for ax, (label, mask) in zip(axes, groups):
        counts, xedges, yedges = np.histogram2d(
            health_score_np[mask], dose[mask], bins=[hs_edges, dose_edges]
        )
        im = ax.pcolormesh(xedges, yedges, counts.T, norm=LogNorm(vmin=1, vmax=counts.max()), cmap="viridis")
        ax.set_xlabel("Health score (severity)")
        ax.set_ylabel("Fluid dose (mean action)")
        ax.set_title(f"{label}\n(n={mask.sum()})", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Count (log scale)")

    fig.suptitle("Fluid dose vs. severity, by vasopressor status (log-scale density)",
                 fontsize=13, fontweight="bold", y=1.03)
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