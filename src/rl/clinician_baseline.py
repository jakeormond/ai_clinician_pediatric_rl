"""
Clinician baseline dosing statistics -- the reference point for every
policy-vs-clinician comparison downstream (extreme-alpha sensitivity
checks, Stage 3 final policy comparison, SARSA cross-evaluation writeups).

No model involved: this is purely a property of the logged data.
"""

import numpy as np
import torch


def compute_clinician_baseline(clinician_actions, zero_threshold=1e-4):
    """
    clinician_actions: (n_timesteps, 2) tensor/array, columns = [fluid, vaso]

    Returns a dict of {fluid,vaso}_{fire_rate,mean_dose,mean_nonzero_dose},
    matching the exact stat names used in run_action_distribution_sweep so
    the two can be plotted/reported side by side directly.
    """
    actions_np = clinician_actions.cpu().numpy() if torch.is_tensor(clinician_actions) else np.asarray(clinician_actions)
    stats = {}
    for dim, name in [(0, "fluid"), (1, "vaso")]:
        dose = actions_np[:, dim]
        fired = dose > zero_threshold
        stats[f"{name}_fire_rate"] = fired.mean()
        stats[f"{name}_mean_dose"] = dose.mean()
        stats[f"{name}_mean_nonzero_dose"] = dose[fired].mean() if fired.any() else np.nan
    return stats


def print_clinician_baseline(stats, label=""):
    print(f"=== Clinician baseline{f' ({label})' if label else ''} ===")
    for name in ["fluid", "vaso"]:
        print(f"  {name}: rate={stats[f'{name}_fire_rate']*100:.1f}%, "
              f"mean={stats[f'{name}_mean_dose']:.4f}, "
              f"nonzero_mean={stats[f'{name}_mean_nonzero_dose']:.4f}")


if __name__ == "__main__":
    # val -- reference for the extreme-alpha sweeps and any Stage 2-era analysis
    clinician_actions_val = processors['val'].rl_dataset["actions"]
    clinician_baseline_val = compute_clinician_baseline(clinician_actions_val)
    print_clinician_baseline(clinician_baseline_val, label="val")

    # test -- reference for final Stage 3 reported results
    clinician_actions_test = processors['test'].rl_dataset["actions"]
    clinician_baseline_test = compute_clinician_baseline(clinician_actions_test)
    print_clinician_baseline(clinician_baseline_test, label="test")