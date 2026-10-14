"""
src/rl/dose_histogram_by_alpha.py

Distribution of recommended doses across the CQL alpha sweep, averaged across seeds. Run on VAL.

Models and data come from src.rl.analysis (runs / models / data); this file holds only the
histogram analysis.

  actions, clin = collect_actions(cql_models, store, split="val")   # once; reused by every plot below
  plot_dose_histograms(actions, clin, dim=0, dim_name="Fluid", alphas_to_show=[...])
  plot_dose_histograms_at_clinician_zero(actions, clin, dim=1, dim_name="Vaso", alphas_to_show=[...])

Averaging. Each seed's actor gives its own histogram, computed on shared bins and normalised to
fractions; the plotted line is the MEAN of those per-seed histograms and the band the min-max
across seeds. Recommended doses are never averaged across seeds at a timestep (that would create
doses no policy recommends and hide bimodal 0-or-1 behaviour).

All seeds cover the same patient-hours (their rows and clinician actions are identical, which
collect_actions checks), so one clinician reference and one clinician-zero mask serve every seed.

Dims: 0 = fluid, 1 = vasopressor. Doses are in [0, 1]; "nonzero" means > data.NONZERO.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.rl.analysis import data as D
from src.rl.evaluate_cql_alpha_sweep import actor_actions, _models_by_alpha

CLIN_COLOR = "#7f8c8d"
MODEL_COLOR = "#2471a3"
NONZERO = D.NONZERO


# =============================================================================
# Collecting actions
# =============================================================================

def collect_actions(cql_models, store, split="val", stochastic=False, sample_seed=0
                    ) -> Tuple[Dict[float, Dict[int, np.ndarray]], np.ndarray]:
    """
    Actions of every (alpha, seed) actor on `split`, plus the clinician's actions.

    cql_models: load_study_models(study) for a study grouped by cql_alpha.
    Returns ({alpha: {seed: (N, 2) actions}}, clinician (N, 2)). Raises if the clinician actions
    differ between seeds (the shared clinician reference would then be wrong).
    """
    actions: Dict[float, Dict[int, np.ndarray]] = {}
    clin: Optional[np.ndarray] = None
    for alpha, seeds in _models_by_alpha(cql_models).items():
        actions[alpha] = {}
        for seed in sorted(seeds):
            m = seeds[seed]
            p = store.processor(m, split)
            a_clin = D.actions(p)
            if clin is None:
                clin = a_clin
            elif clin.shape != a_clin.shape or not np.allclose(clin, a_clin):
                raise ValueError(f"{m.record.label}: clinician actions differ from the first run's; "
                                 f"seeds are not row-aligned")
            actions[alpha][seed] = actor_actions(m, D.states(p), stochastic=stochastic, sample_seed=sample_seed)
    return actions, clin


# =============================================================================
# Histograms
# =============================================================================

def _fractions(x: np.ndarray, edges: np.ndarray, denom: Optional[int] = None) -> np.ndarray:
    """Histogram counts divided by denom (default: len(x))."""
    denom = len(x) if denom is None else denom
    if denom == 0:
        return np.zeros(len(edges) - 1)
    return np.histogram(x, bins=edges)[0] / denom


def seed_averaged_histogram(actions_by_seed: Dict[int, np.ndarray], dim: int, edges: np.ndarray,
                            mask: Optional[np.ndarray] = None, normalize_by: str = "nonzero"
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Histogram of NONZERO recommended doses per seed, then mean / min / max across seeds.

    mask: optional boolean row selector (e.g. clinician-zero rows), shared by all seeds.
    normalize_by: "nonzero" -> fractions of the nonzero doses (shape of the nonzero distribution);
                  "all"     -> fractions of all selected rows (bars show how much mass is nonzero).
    """
    if normalize_by not in ("nonzero", "all"):
        raise ValueError("normalize_by must be 'nonzero' or 'all'")
    hists = []
    for a in actions_by_seed.values():
        x = a[:, dim] if mask is None else a[mask, dim]
        nz = x[x > NONZERO]
        hists.append(_fractions(nz, edges, denom=len(x) if normalize_by == "all" else None))
    h = np.asarray(hists)
    return h.mean(0), h.min(0), h.max(0)


def _fire_rates(actions_by_seed, dim, mask=None) -> np.ndarray:
    out = []
    for a in actions_by_seed.values():
        x = a[:, dim] if mask is None else a[mask, dim]
        out.append((x > NONZERO).mean())
    return np.asarray(out)


# =============================================================================
# Plots
# =============================================================================

def _select_alphas(actions, alphas_to_show) -> list:
    if alphas_to_show is None:
        return sorted(actions)
    missing = [a for a in alphas_to_show if a not in actions]
    if missing:
        print(f"Note: alphas not in the sweep, skipped: {missing}")
    return [a for a in alphas_to_show if a in actions]


def _grid(n, ncols):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes


def plot_dose_histograms(actions, clinician, dim, dim_name, alphas_to_show: Optional[Iterable[float]] = None,
                         n_bins=20, show_seed_range=True, ncols=3, save_path=None):
    """
    One panel per alpha: distribution of NONZERO recommended doses (fraction of the nonzero doses
    per bin), mean across seeds (line) with min-max across seeds (band), over the clinician's
    nonzero distribution (grey). The panel title gives how often the model fires (dose > 0).
    """
    alphas = _select_alphas(actions, alphas_to_show)
    edges = np.linspace(0, 1, n_bins + 1)
    clin_x = clinician[:, dim]
    clin_hist = _fractions(clin_x[clin_x > NONZERO], edges)
    clin_fire = (clin_x > NONZERO).mean()

    fig, axes = _grid(len(alphas), ncols)
    for ax, alpha in zip(axes.flat, alphas):
        mean, lo, hi = seed_averaged_histogram(actions[alpha], dim, edges)
        ax.stairs(clin_hist, edges, fill=True, color=CLIN_COLOR, alpha=0.35, linewidth=0,
                  label=f"clinician (fires {clin_fire * 100:.1f}%)")
        if show_seed_range:
            ax.stairs(hi, edges, baseline=lo, fill=True, color=MODEL_COLOR, alpha=0.2, linewidth=0)
        ax.stairs(mean, edges, color=MODEL_COLOR, linewidth=1.8, label="CQL, mean of seeds")
        fire = _fire_rates(actions[alpha], dim)
        ax.set_title(f"alpha={alpha:g}   fires {fire.mean() * 100:.1f}% [{fire.min() * 100:.1f}-{fire.max() * 100:.1f}]",
                     fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{dim_name} dose")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of nonzero doses")
    axes.flat[0].legend(frameon=False, fontsize=8)
    n_seeds = len(next(iter(actions.values())))
    fig.suptitle(f"{dim_name}: distribution of nonzero recommended doses by alpha "
                 f"(mean across {n_seeds} seeds; band = min-max)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


def plot_dose_histograms_at_clinician_zero(actions, clinician, dim, dim_name,
                                           alphas_to_show: Optional[Iterable[float]] = None,
                                           n_bins=20, show_seed_range=True, ncols=3, save_path=None):
    """
    At the states where the clinician gave NO dose: does the model ever recommend one, and how big?
    One panel per alpha. Bars are fractions of ALL those states (so their total height is the
    share of the clinician-zero states where the model fires); mean across seeds with min-max band.
    """
    alphas = _select_alphas(actions, alphas_to_show)
    edges = np.linspace(0, 1, n_bins + 1)
    mask = clinician[:, dim] <= NONZERO

    fig, axes = _grid(len(alphas), ncols)
    for ax, alpha in zip(axes.flat, alphas):
        mean, lo, hi = seed_averaged_histogram(actions[alpha], dim, edges, mask=mask, normalize_by="all")
        if show_seed_range:
            ax.stairs(hi, edges, baseline=lo, fill=True, color=MODEL_COLOR, alpha=0.2, linewidth=0)
        ax.stairs(mean, edges, color=MODEL_COLOR, linewidth=1.8)
        fire = _fire_rates(actions[alpha], dim, mask=mask)
        ax.set_title(f"alpha={alpha:g}   fires {fire.mean() * 100:.1f}% [{fire.min() * 100:.1f}-{fire.max() * 100:.1f}]",
                     fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{dim_name} dose")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of clinician-zero states")
    n_seeds = len(next(iter(actions.values())))
    fig.suptitle(f"{dim_name}: model dose where the clinician gave none ({int(mask.sum())} states; "
                 f"mean across {n_seeds} seeds; band = min-max)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig