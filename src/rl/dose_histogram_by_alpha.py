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

Beyond the histograms, dose_distribution_metrics() / plot_dose_metrics() reduce each (alpha, seed) to a few numbers
(distance of the nonzero dose distribution from the clinician's, shares of very low and very high doses,
fire rate) so the sweep can be read as curves against alpha.

dose_by_severity() / plot_dose_by_severity() split the same quantities by patient severity (health-score
quantile bins), to see where in the severity range the actors differ from the clinician.

Dims: 0 = fluid, 1 = vasopressor. Doses are in [0, 1]; "nonzero" means > data.NONZERO.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# =============================================================================
# Clinician-likeness metrics per (alpha, seed)
# =============================================================================

_U = (np.arange(1000) + 0.5) / 1000          # quantile grid for the Wasserstein distance


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    """Wasserstein-1 distance between two 1-D samples (same units as the doses, so on [0, 1])."""
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(np.abs(np.quantile(a, _U) - np.quantile(b, _U))))


def dose_distribution_metrics(actions, clinician, dim, low_cut=0.05, high_cut=0.9) -> pd.DataFrame:
    """
    One row per (alpha, seed), on the NONZERO recommended doses of dimension `dim`:
      fire_rate   share of all rows with a dose > 0
      w1          Wasserstein-1 distance of the nonzero dose distribution from the clinician's
                  (0 = identical; in dose units, so 0.1 means 0.1 of the full dose range on average)
      low_share   share of nonzero doses below low_cut (the very-low-dose tail)
      high_share  share of nonzero doses above high_cut
      median_nz   median nonzero dose
    """
    clin_nz = clinician[clinician[:, dim] > NONZERO, dim]
    rows = []
    for alpha in sorted(actions):
        for seed in sorted(actions[alpha]):
            x = actions[alpha][seed][:, dim]
            nz = x[x > NONZERO]
            rows.append({"alpha": alpha, "seed": seed, "fire_rate": float((x > NONZERO).mean()),
                         "w1": _w1(nz, clin_nz), "low_share": float((nz < low_cut).mean()) if len(nz) else np.nan,
                         "high_share": float((nz > high_cut).mean()) if len(nz) else np.nan,
                         "median_nz": float(np.median(nz)) if len(nz) else np.nan})
    return pd.DataFrame(rows)


def clinician_reference(clinician, dim, low_cut=0.05, high_cut=0.9) -> dict:
    """The same metrics for the clinician (w1 = 0 by definition)."""
    x = clinician[:, dim]
    nz = x[x > NONZERO]
    return {"fire_rate": float((x > NONZERO).mean()), "w1": 0.0, "low_share": float((nz < low_cut).mean()),
            "high_share": float((nz > high_cut).mean()), "median_nz": float(np.median(nz))}


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-alpha mean / min / max across seeds for every metric (columns: (metric, stat))."""
    return df.groupby("alpha")[["fire_rate", "w1", "low_share", "high_share", "median_nz"]].agg(["mean", "min", "max"])


def plot_dose_metrics(df, clinician, dim, dim_name, low_cut=0.05, high_cut=0.9, log_scale=True,
                      alpha_range=None, save_path=None):
    """
    The metrics of dose_distribution_metrics against alpha: mean across seeds (line), min-max band,
    and the clinician's value as a dashed line. alpha=0 is dropped on a log axis.
    """
    if alpha_range is not None:
        df = df[(df.alpha >= alpha_range[0]) & (df.alpha <= alpha_range[1])]
    if log_scale:
        df = df[df.alpha > 0]
    ref = clinician_reference(clinician, dim, low_cut, high_cut)
    summ = summarize_metrics(df)
    panels = [("fire_rate", "share of rows with a dose", 100), ("w1", "distance from clinician (W1, dose units)", 1),
              ("low_share", f"share of nonzero doses < {low_cut:g}", 100),
              ("high_share", f"share of nonzero doses > {high_cut:g}", 100)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    x = summ.index.values
    for ax, (col, label, scale) in zip(axes, panels):
        mean, lo, hi = (summ[(col, s)].values * scale for s in ("mean", "min", "max"))
        line, = ax.plot(x, mean, marker="o", markersize=3.5, color=MODEL_COLOR)
        ax.fill_between(x, lo, hi, color=MODEL_COLOR, alpha=0.2, linewidth=0)
        ax.axhline(ref[col] * scale, color=CLIN_COLOR, linestyle="--", linewidth=1.2, label="clinician")
        ax.set_ylabel(label + (" (%)" if scale == 100 else ""), fontsize=9)
        ax.set_xlabel(r"cql_$\alpha$" + (" (log)" if log_scale else ""))
        if log_scale:
            ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    n_seeds = int(df.groupby("alpha")["seed"].nunique().min())
    fig.suptitle(f"{dim_name}: how the recommended dose distribution changes with alpha "
                 f"(mean across {n_seeds} seeds; band = min-max)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig


# =============================================================================
# Dose distribution by severity bin
# =============================================================================

def collect_severity(cql_models, store, split="val") -> np.ndarray:
    """
    Health score at every RL row of `split`. It comes from the shared (seed-averaged) reward, so it
    is identical for every seed; this raises if it is not.
    """
    hs = None
    for alpha, seeds in _models_by_alpha(cql_models).items():
        for seed in sorted(seeds):
            h = D.health_score(store.processor(seeds[seed], split))[0]
            if hs is None:
                hs = h
            elif h.shape != hs.shape or not np.allclose(h, hs, equal_nan=True):
                raise ValueError(f"alpha {alpha} seed {seed}: health score differs from the first run's")
    return hs


def severity_bins(hs: np.ndarray, n_bins: int = 5) -> Tuple[np.ndarray, list]:
    """Equal-count (quantile) bins of the health score. Returns (bin index per row, labels)."""
    edges = np.unique(np.nanquantile(hs, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, hs, side="right") - 1, 0, len(edges) - 2)
    idx = np.where(np.isnan(hs), -1, idx)
    labels = [f"{edges[i]:.2f} to {edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    return idx, labels


def _bin_metrics(x: np.ndarray, clin_nz: np.ndarray, low_cut: float) -> dict:
    nz = x[x > NONZERO]
    return {"n": len(x), "fire_rate": float((x > NONZERO).mean()), "mean_dose": float(x.mean()),
            "median_nz": float(np.median(nz)) if len(nz) else np.nan,
            "low_share": float((nz < low_cut).mean()) if len(nz) else np.nan,
            "w1": _w1(nz, clin_nz)}


def dose_by_severity(actions, clinician, hs, dim, n_bins=5, low_cut=0.05) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    The same quantities as dose_distribution_metrics, within health-score quantile bins.
    Returns (model table: one row per alpha, seed, bin; clinician table: one row per bin; bin labels).
    Columns: n, fire_rate, mean_dose (over all rows, zeros included), median_nz, low_share, w1
    (distance of the bin's nonzero doses from the clinician's in the same bin).
    """
    bidx, labels = severity_bins(hs, n_bins)
    rows, clin_rows = [], []
    for b in range(len(labels)):
        m = bidx == b
        x = clinician[m, dim]
        clin_rows.append({"bin": b, **_bin_metrics(x, x[x > NONZERO], low_cut), "w1": 0.0})
    clin_df = pd.DataFrame(clin_rows)
    for alpha in sorted(actions):
        for seed in sorted(actions[alpha]):
            a = actions[alpha][seed]
            for b in range(len(labels)):
                m = bidx == b
                c = clinician[m, dim]
                rows.append({"alpha": alpha, "seed": seed, "bin": b,
                             **_bin_metrics(a[m, dim], c[c > NONZERO], low_cut)})
    return pd.DataFrame(rows), clin_df, labels


def plot_dose_by_severity(df, clin_df, labels, dim_name, alphas_to_show, low_cut=0.05, save_path=None):
    """
    Mean across seeds (line) and min-max (band) of each metric across severity bins, one colour per
    alpha, the clinician in grey dashed. x axis: health-score quantile bins in ascending order.
    """
    alphas = [a for a in alphas_to_show if a in set(df.alpha)]
    panels = [("fire_rate", "share of rows with a dose"), ("mean_dose", "mean dose (zeros included)"),
              ("median_nz", "median nonzero dose"), ("low_share", f"share of nonzero doses < {low_cut:g}")]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    x = np.arange(len(labels))
    for ax, (col, label) in zip(axes, panels):
        for i, a in enumerate(alphas):
            g = df[df.alpha == a].groupby("bin")[col]
            mean, lo, hi = g.mean().values, g.min().values, g.max().values
            c = cmap(i / max(len(alphas) - 1, 1))
            ax.plot(x, mean, marker="o", markersize=3.5, color=c, label=f"{a:g}")
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
        ax.plot(x, clin_df.sort_values("bin")[col].values, color=CLIN_COLOR, linestyle="--", linewidth=2,
                marker="s", markersize=3.5, label="clinician")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i + 1}" for i in x])
        ax.set_xlabel("severity bin (health-score quantile, ascending)")
        ax.set_ylabel(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(title=r"cql_$\alpha$", frameon=False, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    n_seeds = int(df.groupby("alpha")["seed"].nunique().min())
    fig.suptitle(f"{dim_name} by severity bin (mean across {n_seeds} seeds; band = min-max). "
                 f"Bins: " + "; ".join(f"{i + 1}: {l}" for i, l in enumerate(labels)), fontsize=9, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig