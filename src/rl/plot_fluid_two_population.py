"""
Figure showing the fluid magnitude spike as a genuine two-population
clinical structure: dose vs. severity, colored by whether the patient was
also receiving vasopressors -- illustrating that the "spike" population is
concentrated among sicker, vaso-requiring (shock) patients, consistent
with aggressive fluid resuscitation in shock.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

NO_VASO_COLOR = "#2471a3"
VASO_COLOR = "#c0392b"


def plot_fluid_spike_by_severity_and_vaso(
        f_raw_mean, health_score, clinician_actions,
        vaso_threshold=1e-4, save_path=None,
    ):
    health_score_np = health_score.cpu().numpy() if torch.is_tensor(health_score) else np.asarray(health_score)
    actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    vaso_given = actions_np[:, 1] > vaso_threshold

    dose = 1 / (1 + np.exp(-f_raw_mean))  # sigmoid, back to dose-space for interpretability

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    rng = np.random.default_rng(0)
    n = len(dose)
    if n > 20000:
        idx = rng.choice(n, 20000, replace=False)
    else:
        idx = np.arange(n)

    ax1.scatter(health_score_np[idx][~vaso_given[idx]], dose[idx][~vaso_given[idx]],
                s=4, alpha=0.15, color=NO_VASO_COLOR, label="No vasopressor")
    ax1.scatter(health_score_np[idx][vaso_given[idx]], dose[idx][vaso_given[idx]],
                s=4, alpha=0.3, color=VASO_COLOR, label="Vasopressor given")
    ax1.set_xlabel("Health score (severity)")
    ax1.set_ylabel("Actor fluid dose (mean action)")
    ax1.set_title("(a) Fluid dose vs. severity, by vasopressor status", fontsize=11, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, markerscale=4)
    ax1.spines[["top", "right"]].set_visible(False)

    bins = np.linspace(0, 1, 40)
    ax2.hist(dose[~vaso_given], bins=bins, density=True, alpha=0.6, color=NO_VASO_COLOR,
             label=f"No vasopressor (n={(~vaso_given).sum()})")
    ax2.hist(dose[vaso_given], bins=bins, density=True, alpha=0.6, color=VASO_COLOR,
             label=f"Vasopressor given (n={vaso_given.sum()})")
    ax2.set_xlabel("Actor fluid dose (mean action)")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) Fluid dose distribution, by vasopressor status", fontsize=11, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Fluid dosing separates into two populations aligned with shock status",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


if __name__ == "__main__":
    clinician_actions_test = processors['test'].rl_dataset["actions"]
    fig = plot_fluid_spike_by_severity_and_vaso(
        f_raw_mean, test_health_score, clinician_actions_test,
        save_path="fluid_two_population_by_vaso.png",
    )