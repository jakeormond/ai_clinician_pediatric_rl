"""
src/rl/actor_std_check.py

Are the CQL actors' magnitude stds (and gate probabilities) stuck at a floor or ceiling? Run on VAL.

The hurdle actor outputs, per state: gate probabilities (f_p, v_p), raw magnitude means and magnitude
stds (f_std, v_std), via LoadedModel.actor_outputs(). A std pinned at its lower clamp means the
actor has stopped being stochastic (the magnitude is effectively deterministic); a std pinned at the
upper clamp means it has stopped learning the magnitude. Saturated gate probabilities (near 0 or 1)
are the same problem for the fire/no-fire decision and line up with the "degenerate seed" gate.

  summ = actor_std_summary(cql_models, store, split="val", bounds=None)      # one row per (alpha, seed, dim)
  display(summarize_std(summ, where="fired"))
  plot_std_summary(summ, "Fluid", where="fired")

bounds: (lo, hi) of the actor's std clamp, in the units of f_std / v_std, if you know them (for a clamp on
log-std in [a, b], pass (exp(a), exp(b))). If None, the floor and ceiling are estimated as the smallest and
largest std seen across ALL models and rows, and "at floor / ceiling" means within `tol` of the std range
of them. That estimate is only as good as the sweep's coverage: prefer the true clamp when you have it.

where: "all" = every state; "fired" = states where the gate probability > 0.5 (the only states where
the magnitude is used, so the std that matters).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rl.analysis import data as D
from src.rl.evaluate_cql_alpha_sweep import _models_by_alpha

DIMS = {"Fluid": ("f_std", "f_p"), "Vaso": ("v_std", "v_p")}
MODEL_COLOR = "#2471a3"


def _collect(cql_models, store, split, max_rows, row_seed):
    """{(alpha, seed): {"f_std":..., "v_std":..., "f_p":..., "v_p":...}} on a shared row subsample."""
    out, rows = {}, None
    for alpha, seeds in _models_by_alpha(cql_models).items():
        for seed in sorted(seeds):
            m = seeds[seed]
            s = D.states(store.processor(m, split))
            if rows is None:
                n = len(s)
                rows = np.arange(n) if (max_rows is None or n <= max_rows) else \
                    np.sort(np.random.default_rng(row_seed).choice(n, max_rows, replace=False))
            elif len(s) != n:
                raise ValueError(f"{m.record.label}: {len(s)} rows vs {n}; seeds are not row-aligned")
            o = m.actor_outputs(np.asarray(s)[rows])
            out[(alpha, seed)] = {k: np.asarray(o[k], dtype=np.float32) for k in ("f_std", "v_std", "f_p", "v_p")}
    return out


def actor_std_summary(cql_models, store, split="val", bounds: Optional[Tuple[float, float]] = None,
                      tol: float = 0.01, max_rows: Optional[int] = 20000, row_seed: int = 0,
                      sat: float = 0.01) -> pd.DataFrame:
    """
    One row per (alpha, seed, dim, where). Columns: n, std_mean, std_p05, std_p50, std_p95, std_min, std_max,
    frac_floor, frac_ceil (share of rows with std within tol*range of the floor / ceiling),
    gate_sat_low / gate_sat_high (share of rows with gate probability < sat / > 1 - sat), floor, ceil.
    Rows are subsampled to max_rows (the same rows for every model) to keep memory modest; None = all.
    """
    outs = _collect(cql_models, store, split, max_rows, row_seed)
    if bounds is None:
        lo = min(float(o[k].min()) for o in outs.values() for k in ("f_std", "v_std"))
        hi = max(float(o[k].max()) for o in outs.values() for k in ("f_std", "v_std"))
    else:
        lo, hi = bounds
    span = max(hi - lo, 1e-12)
    rows = []
    for (alpha, seed), o in outs.items():
        for dim, (ks, kp) in DIMS.items():
            for where in ("all", "fired"):
                msk = np.ones(len(o[ks]), bool) if where == "all" else o[kp] > 0.5
                x, p = o[ks][msk], o[kp][msk]
                r = {"alpha": alpha, "seed": seed, "dim": dim, "where": where, "n": int(msk.sum()),
                     "floor": lo, "ceil": hi}
                if len(x):
                    q = np.quantile(x, [0.05, 0.5, 0.95])
                    r.update(std_mean=float(x.mean()), std_p05=float(q[0]), std_p50=float(q[1]), std_p95=float(q[2]),
                             std_min=float(x.min()), std_max=float(x.max()),
                             frac_floor=float((x <= lo + tol * span).mean()), frac_ceil=float((x >= hi - tol * span).mean()),
                             gate_sat_low=float((p < sat).mean()), gate_sat_high=float((p > 1 - sat).mean()))
                rows.append(r)
    return pd.DataFrame(rows)


def summarize_std(summ: pd.DataFrame, where: str = "fired", dim: Optional[str] = None) -> pd.DataFrame:
    """Per alpha (and dim): mean across seeds of the std statistics, and the WORST seed's floor / ceiling share."""
    d = summ[summ["where"] == where]
    if dim is not None:
        d = d[d["dim"] == dim]
    g = d.groupby(["dim", "alpha"])
    return pd.DataFrame({"n_mean": g["n"].mean(), "std_mean": g["std_mean"].mean(),
                         "std_p05": g["std_p05"].mean(), "std_p95": g["std_p95"].mean(),
                         "frac_floor_worst": g["frac_floor"].max(), "frac_ceil_worst": g["frac_ceil"].max(),
                         "gate_sat_low_worst": g["gate_sat_low"].max(), "gate_sat_high_worst": g["gate_sat_high"].max()})


def flag_stuck(summ: pd.DataFrame, where: str = "fired", threshold: float = 0.5, min_rows: int = 200) -> pd.DataFrame:
    """(alpha, seed, dim) runs with more than `threshold` of their rows at the floor or ceiling (and >= min_rows rows)."""
    d = summ[(summ["where"] == where) & (summ["n"] >= min_rows)]
    bad = d[(d.frac_floor > threshold) | (d.frac_ceil > threshold)]
    return bad[["alpha", "seed", "dim", "n", "std_mean", "frac_floor", "frac_ceil"]].sort_values(["dim", "alpha", "seed"])


def plot_std_summary(summ: pd.DataFrame, dim_name: str, where: str = "fired", alpha_range=None, log_scale=True,
                     save_path=None):
    """Mean std, share at floor and share at ceiling against alpha: mean across seeds, min-max band."""
    d = summ[(summ["dim"] == dim_name) & (summ["where"] == where)]
    if alpha_range is not None:
        d = d[(d.alpha >= alpha_range[0]) & (d.alpha <= alpha_range[1])]
    if log_scale:
        d = d[d.alpha > 0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (col, label) in zip(axes, [("std_mean", f"mean {dim_name.lower()} magnitude std"),
                                       ("frac_floor", "share of rows at the std floor"),
                                       ("frac_ceil", "share of rows at the std ceiling")]):
        g = d.groupby("alpha")[col]
        x = g.mean().index.values
        ax.plot(x, g.mean().values, marker="o", markersize=3.5, color=MODEL_COLOR)
        ax.fill_between(x, g.min().values, g.max().values, color=MODEL_COLOR, alpha=0.2, linewidth=0)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel(r"cql_$\alpha$" + (" (log)" if log_scale else ""))
        if log_scale:
            ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{dim_name}: actor magnitude std ({'fired rows' if where == 'fired' else 'all rows'}; "
                 f"mean across seeds, band = min-max)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    return fig