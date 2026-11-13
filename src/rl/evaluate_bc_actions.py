"""
src/rl/evaluate_bc_actions.py

Behaviour-clone action plots on the generic runs / models / data API (the MIMIC counterpart of the CHOC
std / action-distribution / dose-histogram checks). Everything is computed once by collect_bc_outputs()
and reused by the plots:

  out = eval_bc_actions.collect_bc_outputs(bc_models, store, split="test")
  eval_bc_actions.std_summary(out);  eval_bc_actions.plot_std_distributions_by_seed(out)
  eval_bc_actions.action_stats(out); eval_bc_actions.plot_action_distribution(out)
  eval_bc_actions.plot_dose_histograms(out, dim=0, dim_name="Fluid")
  eval_bc_actions.plot_dose_histograms_at_clinician_zero(out, dim=1, dim_name="Vaso")

BC actions are the actor's deterministic action: a drug fires when its gate probability exceeds 0.5, and
the dose is then sigmoid(raw magnitude mean). "exp. fire rate" is the mean gate probability, the
threshold-free version of the fire rate. Dims: 0 = fluid, 1 = vasopressor.
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rl.analysis import data as D
from src.rl.evaluate_fqe_final_seeds import _by_seed

NONZERO = D.NONZERO
CLIN_COLOR = "#7f8c8d"
MODEL_COLOR = "#2a78d6"
DRUGS = (("Fluid", 0, "f"), ("Vaso", 1, "v"))
LOG_STD_MIN, LOG_STD_MAX = -2.0, 1.0          # the BC actor's clamp (read from the actor when it exposes it)


def collect_bc_outputs(bc_models, store, split="test", batch_size=4096) -> dict:
    """
    Per seed: deterministic actions (N, 2), gate probabilities (N, 2), magnitude stds (N, 2); plus the
    clinician's actions (N, 2). Raises if the seeds are not row-aligned.
    """
    by_seed = _by_seed(bc_models)
    seeds = sorted(by_seed)
    out = {"split": split, "seeds": seeds, "actions": {}, "p": {}, "std": {}, "clin": None,
           "log_std_min": LOG_STD_MIN, "log_std_max": LOG_STD_MAX}
    for seed in seeds:
        m = by_seed[seed]
        p = store.processor(m, split)
        a_clin = D.actions(p)
        if out["clin"] is None:
            out["clin"] = a_clin
        elif out["clin"].shape != a_clin.shape or not np.allclose(out["clin"], a_clin):
            raise ValueError(f"{m.record.label}: clinician actions differ from the first run's; seeds are not row-aligned")
        o = m.actor_outputs(D.states(p), batch_size=batch_size) if _accepts_batch(m) else m.actor_outputs(D.states(p))
        out["actions"][seed] = np.asarray(o["actions"])
        out["p"][seed] = np.stack([o["f_p"], o["v_p"]], axis=1)
        out["std"][seed] = np.stack([o["f_std"], o["v_std"]], axis=1)
        out["log_std_min"] = float(getattr(m.actor, "log_std_min", LOG_STD_MIN))
        out["log_std_max"] = float(getattr(m.actor, "log_std_max", LOG_STD_MAX))
    return out


def _accepts_batch(m) -> bool:
    import inspect
    try:
        return "batch_size" in inspect.signature(m.actor_outputs).parameters
    except (TypeError, ValueError):
        return False


# =============================================================================
# Magnitude std: pinned at the clamp?
# =============================================================================

def std_summary(out, floor_tol=1e-3, ceiling_tol=1e-2) -> pd.DataFrame:
    """Per seed and drug: summary of the magnitude std and the share of rows at the floor / ceiling clamp."""
    floor, ceil = np.exp(out["log_std_min"]), np.exp(out["log_std_max"])
    rows = []
    for seed in out["seeds"]:
        for name, col, _ in DRUGS:
            s = out["std"][seed][:, col]
            rows.append({"seed": seed, "drug": name, "mean": s.mean(), "sd": s.std(), "min": s.min(),
                         "median": np.median(s), "max": s.max(),
                         "% at floor": 100 * (np.abs(s - floor) < floor_tol).mean(),
                         "% at ceiling": 100 * (np.abs(s - ceil) < ceiling_tol).mean()})
    print(f"floor std = {floor:.4f} (exp({out['log_std_min']:g})), ceiling std = {ceil:.4f} (exp({out['log_std_max']:g}))")
    return pd.DataFrame(rows).round(4)


def plot_std_distributions_by_seed(out, save_path=None):
    """Violin of the magnitude std per seed (fluid, vaso), with the clamp floor and ceiling."""
    floor, ceil = np.exp(out["log_std_min"]), np.exp(out["log_std_max"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, col, _) in zip(axes, DRUGS):
        data = [out["std"][s][:, col] for s in out["seeds"]]
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(MODEL_COLOR); pc.set_alpha(0.5)
        ax.axhline(floor, color=CLIN_COLOR, linestyle="--", linewidth=1, label="floor")
        ax.axhline(ceil, color=CLIN_COLOR, linestyle=":", linewidth=1, label="ceiling")
        ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels([f"seed {s}" for s in out["seeds"]])
        ax.set_ylabel("magnitude std across rows"); ax.set_title(f"{name} std", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"BC magnitude std per seed ({out['split']})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight"); print(f"Saved to {save_path}")
    return fig


# =============================================================================
# Fire rate / mean dose / mean nonzero dose vs the clinician
# =============================================================================

def _stats(x):
    f = x > NONZERO
    return {"fire_rate": f.mean(), "mean_dose": x.mean(), "mean_nonzero_dose": x[f].mean() if f.any() else np.nan}


def action_stats(out) -> pd.DataFrame:
    """One row per seed and drug (plus a clinician row per drug): fire rate, mean dose, mean nonzero dose, exp. fire rate."""
    rows = []
    for name, col, _ in DRUGS:
        rows.append({"drug": name, "seed": "clinician", **_stats(out["clin"][:, col]), "exp_fire_rate": np.nan})
        for seed in out["seeds"]:
            rows.append({"drug": name, "seed": seed, **_stats(out["actions"][seed][:, col]),
                         "exp_fire_rate": out["p"][seed][:, col].mean()})
    return pd.DataFrame(rows).round(4)


def plot_action_distribution(out, title=None, save_path=None):
    """Fire rate, mean dose and mean nonzero dose per seed (dots, mean as bar) against the clinician (dashed)."""
    df = action_stats(out)
    stats = (("fire_rate", "fire rate"), ("mean_dose", "mean dose (zeros included)"), ("mean_nonzero_dose", "mean nonzero dose"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for r, (name, _, _) in enumerate(DRUGS):
        d = df[df.drug == name]
        clin = d[d.seed == "clinician"].iloc[0]
        mod = d[d.seed != "clinician"]
        for c, (col, label) in enumerate(stats):
            ax = axes[r, c]
            ax.bar([0], [mod[col].mean()], width=0.5, color=MODEL_COLOR, alpha=0.35)
            ax.scatter(np.zeros(len(mod)) + np.linspace(-0.12, 0.12, len(mod)), mod[col], s=22, color=MODEL_COLOR, zorder=3)
            ax.axhline(clin[col], color=CLIN_COLOR, linestyle="--", linewidth=1.5, label="clinician")
            if col == "fire_rate":
                ax.scatter([0.35] * len(mod), mod["exp_fire_rate"], s=22, marker="^", color="#eb6834", zorder=3, label="exp. fire rate (mean gate p)")
            ax.set_xticks([]); ax.set_xlim(-0.5, 0.6)
            ax.set_title(f"{name}: {label}", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(title or f"BC action distribution ({len(out['seeds'])} seeds, {out['split']}); dots = seeds, bar = mean",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight"); print(f"Saved to {save_path}")
    return fig


# =============================================================================
# Dose histograms
# =============================================================================

def _grid(n, ncols):
    ncols = max(1, min(ncols, n)); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes


def plot_dose_histograms(out, dim, dim_name, n_bins=20, ncols=5, save_path=None):
    """Per seed: nonzero BC doses (line) over the clinician's nonzero doses (grey), as fractions of the nonzero doses."""
    edges = np.linspace(0, 1, n_bins + 1)
    cx = out["clin"][:, dim]; cnz = cx[cx > NONZERO]
    chist = np.histogram(cnz, bins=edges)[0] / max(len(cnz), 1)
    fig, axes = _grid(len(out["seeds"]), ncols)
    for ax, seed in zip(axes.flat, out["seeds"]):
        x = out["actions"][seed][:, dim]; nz = x[x > NONZERO]
        ax.stairs(chist, edges, fill=True, color=CLIN_COLOR, alpha=0.35, linewidth=0, label=f"clinician (fires {(cx > NONZERO).mean() * 100:.1f}%)")
        ax.stairs(np.histogram(nz, bins=edges)[0] / max(len(nz), 1), edges, color=MODEL_COLOR, linewidth=1.8, label="BC")
        ax.set_title(f"seed {seed}   fires {(x > NONZERO).mean() * 100:.1f}%", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{dim_name} dose")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of nonzero doses")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{dim_name}: nonzero BC doses vs clinician ({out['split']})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight"); print(f"Saved to {save_path}")
    return fig


def plot_dose_histograms_at_clinician_zero(out, dim, dim_name, n_bins=20, ncols=5, save_path=None):
    """At rows where the clinician gave none: BC's doses per seed, as fractions of ALL those rows (bar total = BC fire rate there)."""
    edges = np.linspace(0, 1, n_bins + 1)
    mask = out["clin"][:, dim] <= NONZERO
    fig, axes = _grid(len(out["seeds"]), ncols)
    for ax, seed in zip(axes.flat, out["seeds"]):
        x = out["actions"][seed][mask, dim]; nz = x[x > NONZERO]
        ax.stairs(np.histogram(nz, bins=edges)[0] / max(len(x), 1), edges, fill=True, color=MODEL_COLOR, alpha=0.35, linewidth=0)
        ax.stairs(np.histogram(nz, bins=edges)[0] / max(len(x), 1), edges, color=MODEL_COLOR, linewidth=1.5)
        ax.set_title(f"seed {seed}   fires {(x > NONZERO).mean() * 100:.1f}%", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{dim_name} dose")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of clinician-zero rows")
    fig.suptitle(f"{dim_name}: BC dose where the clinician gave none ({int(mask.sum())} rows, {out['split']})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight"); print(f"Saved to {save_path}")
    return fig